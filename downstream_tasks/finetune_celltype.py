import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy, F1Score
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch_geometric.utils import from_scipy_sparse_matrix

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import scipy.sparse as sp
def compute_laplacian_pos_enc(edge_index, num_nodes, k=32, device=DEVICE, cache_path=None, dense_threshold=8000):
    if cache_path and os.path.exists(cache_path):
        pe = np.load(cache_path)
        return torch.from_numpy(pe).float()

    row, col = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
    data = np.ones_like(row, dtype=np.float32)
    A = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    A = (A + A.T).maximum(A)

    if num_nodes <= dense_threshold and torch.cuda.is_available():
        A_dense = torch.from_numpy(A.toarray()).float().to(device)
        deg = A_dense.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg + 1e-12, -0.5)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        L = torch.eye(num_nodes, device=device) - D_inv_sqrt @ A_dense @ D_inv_sqrt
        vals, vecs = torch.linalg.eigh(L)
        vecs = vecs[:, 1:k+1].cpu().numpy()
    else:
        L_cpu = sp.csgraph.laplacian(A, normed=True)
        k_eff = min(k + 1, num_nodes - 1)
        vals, vecs = sp.linalg.eigsh(L_cpu, k=k_eff, which='SM', tol=1e-5)
        vecs = vecs[:, 1:k+1]

    if cache_path:
        np.save(cache_path, vecs)
    return torch.from_numpy(vecs).float()

# data_utils.py

import os
import anndata as an
import numpy as np

def load_and_combine_adata(reference_path, query_path, label_key_col, gene_name_col):
    """Loads, standardizes, and concatenates reference and query AnnData objects."""
    print("--- Loading and Combining AnnData objects ---")
    if not os.path.exists(reference_path) or not os.path.exists(query_path):
        raise FileNotFoundError("Reference or Query data not found. Please check paths.")
    adata_ref = an.read_h5ad(reference_path)
    adata_query = an.read_h5ad(query_path)

    adata_ref.obs['celltype'] = adata_ref.obs[label_key_col].astype("category")
    adata_query.obs['celltype'] = adata_query.obs[label_key_col].astype("category")
    adata_ref.var.set_index(adata_ref.var[gene_name_col].astype(str), inplace=True)
    adata_query.var.set_index(adata_query.var[gene_name_col].astype(str), inplace=True)

    combined_adata = an.concat(
        {"reference": adata_ref, "query": adata_query},
        label="source", index_unique=None
    )
    print(f"Combined AnnData created with {combined_adata.n_obs} cells from sources: {combined_adata.obs['source'].unique().tolist()}")
    return combined_adata

def filter_adata_to_genes(combined_adata, graph_gene_names):
    """
    Finds the intersection of genes and returns the filtered AnnData object and
    the indices of the intersecting genes relative to the original graph vocabulary.
    """
    print("--- Filtering AnnData to Graph Vocabulary ---")
    graph_gene_names = graph_gene_names.astype(str)
    combined_adata.var_names = combined_adata.var_names.astype(str)

    var_name_to_index = {g: i for i, g in enumerate(combined_adata.var_names)}
    graph_sel_idx = [] # Indices of intersecting genes in the original graph
    adata_sel_idx = [] # Indices of intersecting genes in the anndata object

    for i, g in enumerate(graph_gene_names):
        j = var_name_to_index.get(g)
        if j is not None:
            graph_sel_idx.append(i)
            adata_sel_idx.append(j)

    if len(graph_sel_idx) == 0:
        raise RuntimeError('No gene overlap between graph and AnnData.')

    adata_filtered = combined_adata[:, adata_sel_idx].copy()
    print(f"Filtered to {len(graph_sel_idx)} overlapping genes.")
    return adata_filtered, graph_sel_idx

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
import torchmetrics

# Minimal subset extracted from pretraining script for reuse in finetuning

PAD_TOKEN_ID_PLACEHOLDER = 10_000  # Will be overwritten at runtime
MASK_TOKEN_ID_PLACEHOLDER = 10_001


def create_scheduler_with_warmup(optimizer, warmup_steps, total_steps, eta_min=1e-6):
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=max(1, total_steps - warmup_steps), eta_min=eta_min)
    return SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])


class GeneExpressionEmbedding(nn.Module):
    def __init__(self, num_genes, go_feature_dim, pos_encoding_dim, embedding_dim, full_vocab_size, pad_token_id):
        super().__init__()
        self.gene_embedding = nn.Embedding(num_genes, embedding_dim)
        self.value_embedding = nn.Embedding(full_vocab_size, embedding_dim, padding_idx=pad_token_id)
        combined_feature_dim = embedding_dim + embedding_dim + go_feature_dim + pos_encoding_dim
        self.feature_projection = nn.Linear(combined_feature_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, gene_ids, expression_values, go_features, pos_enc):
        B = expression_values.size(0)
        gene_e = self.gene_embedding(gene_ids).unsqueeze(0).expand(B, -1, -1)
        val_e = self.value_embedding(expression_values)
        go_f_expanded = go_features.unsqueeze(0).expand(B, -1, -1)
        pos_e_expanded = pos_enc.unsqueeze(0).expand(B, -1, -1)
        combined_features = torch.cat([gene_e, val_e, go_f_expanded, pos_e_expanded], dim=-1)
        x = self.feature_projection(combined_features)
        return self.norm(x)


class GraphTransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        from torch_geometric.nn import GATv2Conv
        edge_feature_dim = 16
        self.attention = GATv2Conv(
            in_channels=embed_dim,
            out_channels=embed_dim,
            heads=num_heads,
            concat=True,
            dropout=dropout,
            edge_dim=edge_feature_dim
        )
        self.edge_projector = nn.Linear(1, edge_feature_dim)
        self.linear = nn.Linear(embed_dim * num_heads, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, edge_index, edge_attr):
        projected_edge_attr = self.edge_projector(edge_attr.unsqueeze(-1))
        attn_out = self.linear(self.attention(x, edge_index, edge_attr=projected_edge_attr))
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


def make_batched_edge_index(edge_index, num_genes, batch_size, device=None):
    base = edge_index.long()
    rows, cols = [], []
    for b in range(batch_size):
        offset = b * num_genes
        rows.append(base[0] + offset)
        cols.append(base[1] + offset)
    rows = torch.cat(rows, dim=0)
    cols = torch.cat(cols, dim=0)
    batched = torch.stack([rows, cols], dim=0)
    if device is not None:
        batched = batched.to(device)
    return batched


def make_batched_edge_attr(edge_attr, batch_size, device=None):
    batched = edge_attr.repeat(batch_size)
    if device is not None:
        batched = batched.to(device)
    return batched


class HybridGraphTransformer(nn.Module):
    def __init__(self, num_local_layers, num_global_layers, embed_dim, num_heads, dropout, max_batch_size=32):
        super().__init__()
        self.local_encoder = nn.ModuleList(
            [GraphTransformerLayer(embed_dim, num_heads, dropout) for _ in range(num_local_layers)]
        )
        global_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 2,
            dropout=dropout, batch_first=True
        )
        self.global_encoder = nn.TransformerEncoder(global_layer, num_layers=num_global_layers)
        self.final_norm = nn.LayerNorm(embed_dim)
        self._batched_edge_cache = {}
        self._max_batch_size = max_batch_size

    def forward(self, gene_feats, cls_tokens, edge_index, edge_attr, batch_size):
        B, G, D = gene_feats.shape
        assert B == batch_size, 'Batch size mismatch'
        x_flat = gene_feats.reshape(B * G, D)
        cache_key = (batch_size, edge_index.device)
        if cache_key not in self._batched_edge_cache:
            batched_edge_index = make_batched_edge_index(edge_index, G, batch_size, device=edge_index.device)
            batched_edge_attr = make_batched_edge_attr(edge_attr, batch_size, device=edge_attr.device)
            if batch_size <= self._max_batch_size:
                self._batched_edge_cache[cache_key] = (batched_edge_index, batched_edge_attr)
        else:
            batched_edge_index, batched_edge_attr = self._batched_edge_cache[cache_key]
        x_local_flat = x_flat
        for layer in self.local_encoder:
            x_local_flat = layer(x_local_flat, batched_edge_index, batched_edge_attr)
        x_local_genes = x_local_flat.reshape(B, G, D)
        full_sequence_input = torch.cat([cls_tokens, x_local_genes], dim=1)
        processed_sequence = full_sequence_input
        if self.training:
            for layer in self.global_encoder.layers:
                processed_sequence = torch.utils.checkpoint.checkpoint(
                    layer, processed_sequence, use_reentrant=False
                )
        else:
            processed_sequence = self.global_encoder(processed_sequence)

        out = self.final_norm(processed_sequence)
        return out


class FoundationalGeneTransformer(pl.LightningModule):
    def __init__(self, config, num_genes, go_feature_dim, pos_encoding_dim, edge_index, edge_attr, go_features, pos_enc, full_vocab_size, pad_token_id, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        config = self.hparams.config
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_attr', edge_attr)
        self.register_buffer('go_features', go_features)
        self.register_buffer('pos_enc', pos_enc)
        self.register_buffer('gene_ids', torch.arange(num_genes))

        # --- THIS IS THE FIX ---
        # We now pass the correct arguments in the correct order to GeneExpressionEmbedding
        self.embedding_layer = GeneExpressionEmbedding(
            num_genes=self.hparams.num_genes,
            go_feature_dim=self.hparams.go_feature_dim,
            pos_encoding_dim=self.hparams.pos_encoding_dim,
            embedding_dim=config['embedding_dim'],
            full_vocab_size=self.hparams.full_vocab_size,
            pad_token_id=self.hparams.pad_token_id
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, config['embedding_dim']))
        self.encoder = HybridGraphTransformer(
            num_local_layers=config['num_local_layers'],
            num_global_layers=config['num_global_layers'],
            embed_dim=config['embedding_dim'],
            num_heads=config['num_heads'],
            dropout=config['dropout']
        )

    def forward(self, x_values):
        batch_size = x_values.size(0)

        # 1. Get the initial embeddings for all genes in the batch
        # This correctly has the shape [batch_size, num_genes, embedding_dim]
        full_embeddings = self.embedding_layer(
            self.gene_ids, x_values, self.go_features, self.pos_enc
        )

        # 2. Prepare the CLS token for the batch
        cls_token_expanded = self.cls_token.expand(batch_size, -1, -1)

        # 3. --- THIS IS THE CORRECTED CALL ---
        # Call the encoder with the correct separate arguments as defined in its signature.
        encoded_features = self.encoder(
            gene_feats=full_embeddings,
            cls_tokens=cls_token_expanded,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            batch_size=batch_size
        )

        # The encoder returns the full sequence [CLS, gene1, gene2, ...].
        # The cell representation is the output corresponding to the CLS token at index 0.
        cell_representation = encoded_features[:, 0]

        # The original pre-training model returned gene embeddings as well.
        # For fine-tuning, we only need the cell representation, so returning None for the first value is fine.
        return None, cell_representation



    def get_cell_embeddings(self, masked_input):
        with torch.no_grad():
            _, cell_repr = self.forward(masked_input)
        return cell_repr

    @torch.no_grad()
    def get_gene_embeddings(self):
        self.eval()
        G = self.hparams.get('num_genes', self.gene_ids.size(0))
        zero_input = torch.full((1, G), self.embedding_layer.value_embedding.padding_idx, dtype=torch.long, device=self.device)
        gene_out, _ = self.forward(zero_input)
        return gene_out.squeeze(0).cpu().numpy()

    def configure_optimizers(self):
        cfg = self.hparams['config']
        optimizer = AdamW(self.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
        total_steps = cfg.get('total_steps', 10_000)
        scheduler = create_scheduler_with_warmup(optimizer, warmup_steps=cfg.get('warmup_steps', 1000), total_steps=total_steps)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}


def load_foundation_from_legacy_checkpoint(
    config, num_genes, go_feature_dim, pos_encoding_dim, edge_index, edge_attr,
    go_features, pos_enc, full_vocab_size, pad_token_id, checkpoint_path, map_location='cpu'
):
    """
    Manually constructs a FoundationalGeneTransformer with dimensions from a legacy checkpoint,
    loads the weights, and then adapts the model's layers and buffers to match new data dimensions.
    """
    print(f"--- Manually loading and adapting legacy checkpoint: {checkpoint_path} ---")

    # 1. Load the checkpoint and inspect its state_dict
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    state_dict = ckpt['state_dict']

    # 2. Infer old dimensions from the state_dict to build a model that matches the checkpoint
    old_gene_emb_weight = state_dict['embedding_layer.gene_embedding.weight']
    old_num_genes = old_gene_emb_weight.shape[0]

    # Infer feature dimensions, trying to use hparams from the checkpoint if available
    if 'hparams' in ckpt and 'go_feature_dim' in ckpt['hparams']:
        old_go_dim = ckpt['hparams']['go_feature_dim']
        old_pos_dim = ckpt['hparams']['pos_encoding_dim']
    else: # Fallback to inferring from tensor shapes in the state_dict
        old_go_dim = state_dict['go_features'].shape[1]
        old_pos_dim = state_dict['pos_enc'].shape[1]

    # 3. Build the model shell with the CHECKPOINT's dimensions.
    # We must use the buffers from the state_dict for instantiation to ensure shapes match perfectly.
    model = FoundationalGeneTransformer(
        config=config,
        num_genes=old_num_genes,
        go_feature_dim=old_go_dim,
        pos_encoding_dim=old_pos_dim,
        edge_index=state_dict['edge_index'],
        edge_attr=state_dict['edge_attr'],
        go_features=state_dict['go_features'],
        pos_enc=state_dict['pos_enc'],
        full_vocab_size=full_vocab_size,
        pad_token_id=pad_token_id
    )

    # 4. Load the weights into the perfectly matched model shell.
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print("Legacy checkpoint loaded into shell with some incompatibilities (often expected):")
        print(f"  Missing keys: {incompatible.missing_keys}")
        print(f"  Unexpected keys: {incompatible.unexpected_keys}")
    else:
        print("Legacy checkpoint weights loaded into model shell successfully.")


    # --- ADAPTATION STEP ---
    print(f"Adapting model from {old_num_genes} genes to {num_genes} genes...")

    # 5. Resize the gene embedding layer by creating a new one and copying the old weights.
    old_embedding = model.embedding_layer.gene_embedding
    new_embedding = nn.Embedding(num_genes, config['embedding_dim'])
    # Copy the old weights over the corresponding part of the new layer
    with torch.no_grad():
        new_embedding.weight[:old_num_genes, :] = old_embedding.weight
    # Replace the old layer with the new one
    model.embedding_layer.gene_embedding = new_embedding

    # 6. Replace the old graph-related buffers with the new ones passed as arguments.
    model.register_buffer('edge_index', edge_index)
    model.register_buffer('edge_attr', edge_attr)
    model.register_buffer('go_features', go_features)
    model.register_buffer('pos_enc', pos_enc)
    model.register_buffer('gene_ids', torch.arange(num_genes))

    # 7. Update hyperparameters stored in the model for consistency.
    model.hparams.num_genes = num_genes
    model.hparams.go_feature_dim = go_feature_dim
    model.hparams.pos_encoding_dim = pos_encoding_dim

    print("Model adaptation complete.")
    return model



REFERENCE_ADATA_PATH = '/home/elenamuia/tesi/downstream_tasks/ms/data/c_data.h5ad'
QUERY_ADATA_PATH = '/home/elenamuia/tesi/downstream_tasks/ms/data/filtered_ms_adata.h5ad'
MS_LABEL_COL = "Factor Value[inferred cell type - authors labels]"
MS_GENE_COL = "gene_name"

PRETRAINED_CKPT = '/home/elenamuia/tesi/checkpoints/check_w_feat/best-model-loss-epoch=80-val_loss=1.4775.ckpt'
GENE_NAMES_PATH = '/home/elenamuia/tesi/outputs/graph_gene_names_4k.npy'
GO_FEATURES_PATH = '/home/elenamuia/tesi/outputs/gene_go_features_4k.npy'
GRAPH_PATH = '/home/elenamuia/tesi/outputs/A_context_multimodal_4k.npz'
POS_ENC_CACHE = '/home/elenamuia/tesi/outputs/laplacian_pos_enc_32.npy'
EXPR_BINS_PATH = '/home/elenamuia/tesi/outputs/expression_bins.npy'

LABEL_KEY = 'celltype' # Standardized name after loading
OUTPUT_DIR = '/home/elenamuia/tesi/downstream_tasks/checkpoints_celltype_ms'
os.makedirs(OUTPUT_DIR, exist_ok=True)

FREEZE_BACKBONE = False
BATCH_SIZE = 32
MAX_EPOCHS = 50
LR_HEAD = 1e-5
BALANCE_WEIGHTS = True

class CellTypeDataset(Dataset):
    def __init__(self, adata, indices, bins):
        self.adata = adata
        self.indices = indices
        self.bins = bins
        self.expr = adata.X[indices].toarray() if hasattr(adata.X, 'toarray') else adata.X[indices]
        self.binned = np.digitize(self.expr, bins=bins, right=True).astype(np.int64)
        labels = adata.obs[LABEL_KEY].astype('category')
        self.label_codes = labels.cat.codes.values[indices].astype(np.int64)
        self.num_classes = len(labels.cat.categories)
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx): return torch.from_numpy(self.binned[idx]).long(), torch.tensor(self.label_codes[idx], dtype=torch.long)

class CellTypeDataModule(pl.LightningDataModule):
    def __init__(self, adata, bins, train_idx, val_idx, batch_size=32):
        super().__init__()
        self.adata = adata
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.bins = bins
        self.batch_size = batch_size
    def setup(self, stage=None):
        self.train_ds = CellTypeDataset(self.adata, self.train_idx, self.bins)
        self.val_ds = CellTypeDataset(self.adata, self.val_idx, self.bins)
    def train_dataloader(self): return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4)
    def val_dataloader(self): return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=4)

class CellTypeFinetuner(pl.LightningModule):
    def __init__(self, backbone: FoundationalGeneTransformer, num_classes, model_gene_indices, lr_head=1e-3, freeze_backbone=True, class_weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=['backbone', 'class_weights'])
        self.backbone = backbone
        self.register_buffer('model_gene_indices', model_gene_indices)

        emb_dim = self.backbone.hparams.config['embedding_dim']
        self.classifier = nn.Linear(emb_dim, num_classes)
        if freeze_backbone: self.backbone.freeze()

        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.f1_metric = F1Score(task='multiclass', num_classes=num_classes, average='macro')

    def forward(self, x_subset):
        batch_size = x_subset.size(0)
        num_total_genes = self.backbone.hparams.num_genes
        pad_token_id = self.backbone.hparams.pad_token_id

        full_input = torch.full((batch_size, num_total_genes), fill_value=pad_token_id, device=x_subset.device, dtype=x_subset.dtype)
        full_input[:, self.model_gene_indices] = x_subset

        with torch.set_grad_enabled(not self.hparams.freeze_backbone):
            _, cell_repr = self.backbone(full_input)

        return self.classifier(cell_repr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss)
        self.log('train_f1', self.f1_metric(logits.argmax(1), y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_f1', self.f1_metric(logits.argmax(1), y), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_f1', self.f1_metric(logits.argmax(1), y), prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr_head)

def main():
    # --- Step 1: Load ORIGINAL graph components and full model as a weight source ---
    print("--- Loading ORIGINAL graph components for backbone construction ---")
    graph_gene_names = np.load(GENE_NAMES_PATH, allow_pickle=True)
    num_original_genes = len(graph_gene_names)
    go_features_full = torch.from_numpy(np.load(GO_FEATURES_PATH)).float()
    adj_full = sp.load_npz(GRAPH_PATH)
    edge_index_full, edge_attr_full = from_scipy_sparse_matrix(adj_full)
    edge_attr_full = edge_attr_full.float()
    pos_enc_full = torch.from_numpy(np.load(POS_ENC_CACHE)).float()

    config = {'embedding_dim': 128, 'hidden_channels': 256, 'num_heads': 2, 'num_local_layers': 2, 'num_global_layers': 4, 'dropout': 0.1}
    bins = np.load(EXPR_BINS_PATH) if os.path.exists(EXPR_BINS_PATH) else None
    vocab_size = (len(bins) + 1) if bins is not None else 512
    pad_token_id = vocab_size

    # Load the full pre-trained model. We will use it as a source of weights.
    full_backbone_weight_source = load_foundation_from_legacy_checkpoint(
        config=config, num_genes=num_original_genes,
        go_feature_dim=go_features_full.shape[1], pos_encoding_dim=pos_enc_full.shape[1],
        edge_index=edge_index_full, edge_attr=edge_attr_full, go_features=go_features_full, pos_enc=pos_enc_full,
        full_vocab_size=vocab_size + 2, pad_token_id=pad_token_id,
        checkpoint_path=PRETRAINED_CKPT
    )
    full_backbone_weight_source.eval()


    # --- Step 2: Load AnnData and Define Subgraph ---
    combined_adata = load_and_combine_adata(REFERENCE_ADATA_PATH, QUERY_ADATA_PATH, MS_LABEL_COL, MS_GENE_COL)

    # 2.1. Find the initial intersection ("seed nodes")
    print("--- Finding initial gene overlap ---")
    graph_gene_names_set = set(graph_gene_names)
    adata_var_names_set = set(combined_adata.var_names)
    common_genes = sorted(list(graph_gene_names_set.intersection(adata_var_names_set)))

    if not common_genes:
        raise ValueError("No common genes found. Check gene identifiers (e.g., symbols vs. Ensembl IDs).")

    graph_gene_to_idx = {name: i for i, name in enumerate(graph_gene_names)}
    seed_node_indices = np.array([graph_gene_to_idx[name] for name in common_genes])
    print(f"Found {len(seed_node_indices)} seed genes (initial overlap).")

    # 2.2. Expand the gene set to the 2-hop neighborhood
    print("--- Expanding gene set with 2-hop neighbors ---")
    adj_binary = (adj_full > 0).astype(int).tocsr()
    adj_2_hop = adj_binary @ adj_binary
    one_hop_neighbors = adj_binary[seed_node_indices].sum(axis=0).nonzero()[1]
    two_hop_neighbors = adj_2_hop[seed_node_indices].sum(axis=0).nonzero()[1]
    graph_sel_idx = np.union1d(np.union1d(seed_node_indices, one_hop_neighbors), two_hop_neighbors)
    print(f"Expanded gene set to {len(graph_sel_idx)} nodes for model context.")


    # --- Step 3: Create and Prepare Subgraph Model for Finetuning ---

    # 3.1. Create the SUBGRAPH components for the new model
    num_subgraph_genes = len(graph_sel_idx)
    go_features_sub = go_features_full[graph_sel_idx]
    pos_enc_sub = pos_enc_full[graph_sel_idx]
    adj_full_csr = adj_full.tocsr() # Convert to CSR for slicing
    adj_sub = adj_full_csr[graph_sel_idx, :][:, graph_sel_idx]
    edge_index_sub, edge_attr_sub = from_scipy_sparse_matrix(adj_sub)
    edge_attr_sub = edge_attr_sub.float()

    # 3.2. Create a new, smaller backbone specifically for the subgraph
    print(f"--- Creating a new backbone for the {num_subgraph_genes}-gene subgraph ---")
    finetune_backbone = FoundationalGeneTransformer(
        config=config, num_genes=num_subgraph_genes,
        go_feature_dim=go_features_sub.shape[1], pos_encoding_dim=pos_enc_sub.shape[1],
        edge_index=edge_index_sub, edge_attr=edge_attr_sub, go_features=go_features_sub, pos_enc=pos_enc_sub,
        full_vocab_size=vocab_size + 2, pad_token_id=pad_token_id
    )

    # 3.3. Manually copy relevant weights from the full model to the smaller subgraph model
    print("--- Surgically copying weights from full model to subgraph model ---")
    source_state_dict = full_backbone_weight_source.state_dict()
    target_state_dict = finetune_backbone.state_dict()
    new_state_dict = {}

    for name, param in target_state_dict.items():
        # Copy gene-agnostic layers (those with matching shapes)
        if name in source_state_dict and source_state_dict[name].shape == param.shape:
            new_state_dict[name] = source_state_dict[name].clone()
        else:
            # Buffers and gene-specific layers will be skipped
            pass

    # Manually copy the gene_embedding weights for the selected subgraph genes
    new_state_dict['embedding_layer.gene_embedding.weight'] = \
        source_state_dict['embedding_layer.gene_embedding.weight'][graph_sel_idx].clone()

    # Load the carefully constructed state dictionary. `strict=False` is a good safeguard.
    finetune_backbone.load_state_dict(new_state_dict, strict=False)
    print("Weights copied successfully to the new subgraph backbone.")

    # 3.4. Filter AnnData to only the common genes with expression data
    adata_filtered = combined_adata[:, common_genes].copy()

    # --- Step 4: Stratified Data Splitting ---
    print("\n--- Performing stratified split on the combined dataset ---")
    all_indices = np.arange(adata_filtered.n_obs)
    labels = adata_filtered.obs[LABEL_KEY].cat.codes.values

    # First split: 85% for training+validation, 15% for testing
    train_val_indices, test_indices, train_val_labels, _ = train_test_split(
        all_indices, labels, test_size=0.15, random_state=42, stratify=labels
    )
    # Second split: Split the 85% into train and validation sets (70% train, 15% val of total)
    train_indices, val_indices, _, _ = train_test_split(
        train_val_indices, train_val_labels, test_size=(0.15 / 0.85), random_state=42, stratify=train_val_labels
    )
    print(f"Data split: {len(train_indices)} train, {len(val_indices)} validation, {len(test_indices)} test cells.\n")


    # --- Step 5: Setup the Finetuner and Trainer ---
    # This mapping tells the finetuner where the common genes are located within the new subgraph space.
    subgraph_orig_idx_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(graph_sel_idx)}
    model_gene_indices = torch.tensor([subgraph_orig_idx_map[i] for i in seed_node_indices], dtype=torch.long)

    dm = CellTypeDataModule(adata_filtered, bins, train_idx=train_indices, val_idx=val_indices, batch_size=BATCH_SIZE)
    dm.setup()

    class_weights = None
    if BALANCE_WEIGHTS:
        from collections import Counter
        counts = Counter(dm.train_ds.label_codes)
        w = torch.tensor([sum(counts.values()) / counts.get(i, 1) for i in range(dm.train_ds.num_classes)], dtype=torch.float)
        class_weights = w / w.sum() * dm.train_ds.num_classes

    # Pass the NEW smaller and context-aware backbone to the finetuner
    model = CellTypeFinetuner(
        backbone=finetune_backbone, num_classes=dm.train_ds.num_classes, model_gene_indices=model_gene_indices,
        lr_head=LR_HEAD, freeze_backbone=FREEZE_BACKBONE, class_weights=class_weights
    )

    trainer = pl.Trainer(
        accelerator='gpu', devices=1, max_epochs=MAX_EPOCHS,
        callbacks=[
            ModelCheckpoint(dirpath=OUTPUT_DIR, monitor='val_f1', mode='max', filename='celltype-subgraph-{epoch:02d}-{val_f1:.3f}'),
            EarlyStopping(monitor='val_f1', mode='max', patience=5)
        ], logger=WandbLogger(project='scgt-finetune', name='celltype-ms-subgraph') if 'WANDB_API_KEY' in os.environ else None
    )

    trainer.fit(model, dm)
    
    print("\n--- Evaluating on the stratified test set ---")
    test_ds = CellTypeDataset(adata_filtered, test_indices, bins)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=4)
    trainer.test(dataloaders=test_loader, ckpt_path='best')

if __name__ == '__main__':
    main()