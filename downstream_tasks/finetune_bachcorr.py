import os
# WARNING: Hardcoding API keys is not recommended for shared code.
# Consider using Colab secrets or environment variables.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import torch
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.sparse as sp
import anndata as an

# =====================================================================================
# CONFIGURATION
# =====================================================================================

# --- Data Paths for MS Dataset ---
REFERENCE_ADATA_PATH = '/home/elenamuia/tesi/downstream_tasks/ms/data/c_data.h5ad'
QUERY_ADATA_PATH = '/home/elenamuia/tesi/downstream_tasks/ms/data/filtered_ms_adata.h5ad'
MS_LABEL_COL = "Factor Value[inferred cell type - authors labels]" # Used for stratification
MS_GENE_COL = "gene_name"
BATCH_KEY = 'source' # Use the 'reference'/'query' distinction as the batch signal

# --- Model & Graph Paths ---
PRETRAINED_CKPT = '/home/elenamuia/tesi/checkpoints/check_w_feat/best-model-loss-epoch=80-val_loss=1.4775.ckpt'
GENE_NAMES_PATH = '/home/elenamuia/tesi/outputs/graph_gene_names_4k.npy'
GO_FEATURES_PATH = '/home/elenamuia/tesi/outputs/gene_go_features_4k.npy'
GRAPH_PATH = '/home/elenamuia/tesi/outputs/A_context_multimodal_4k.npz'
POS_ENC_CACHE = '/home/elenamuia/tesi/outputs/laplacian_pos_enc_32.npy'
EXPR_BINS_PATH = '/home/elenamuia/tesi/outputs/expression_bins.npy'
OUTPUT_DIR = '/home/elenamuia/tesi/downstream_tasks/checkpoints_batchcorr_ms_subgraph'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Hyperparameters for Batch Correction Fine-tuning ---
FREEZE_BACKBONE = False
BATCH_SIZE = 32
MAX_EPOCHS = 50
LR = 1e-5          # Lower learning rate for fine-tuning the whole model
LAMBDA_ADV = 0.1   # Weight for the adversarial loss

# =====================================================================================
# UTILITY AND DATA LOADING FUNCTIONS
# =====================================================================================
def load_and_combine_adata(reference_path, query_path, gene_name_col):
    """
    Loads, standardizes gene names, and concatenates reference and query AnnData objects
    for batch correction.
    """
    print("--- Loading and Combining AnnData objects ---")
    adata_ref = an.read_h5ad(reference_path)
    adata_query = an.read_h5ad(query_path)

    # We only need to set the var_names to the gene symbols for matching
    adata_ref.var.set_index(adata_ref.var[gene_name_col].astype(str), inplace=True)
    adata_query.var.set_index(adata_query.var[gene_name_col].astype(str), inplace=True)

    # The label processing lines are removed as they are not needed for this task.

    combined_adata = an.concat(
        {"reference": adata_ref, "query": adata_query},
        label="source",  # This 'source' column will be used as the BATCH_KEY
        index_unique=None
    )
    print(f"Combined AnnData created with {combined_adata.n_obs} cells from sources: {combined_adata.obs['source'].unique().tolist()}")
    return combined_adata

# =====================================================================================
# FOUNDATION MODEL CLASSES (Self-contained)
# =====================================================================================
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
        return self.norm(self.feature_projection(combined_features))

class GraphTransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        from torch_geometric.nn import GATv2Conv
        self.attention = GATv2Conv(embed_dim, embed_dim, heads=num_heads, concat=True, dropout=dropout, edge_dim=16)
        self.edge_projector = nn.Linear(1, 16)
        self.linear = nn.Linear(embed_dim * num_heads, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(dropout))
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, edge_index, edge_attr):
        projected_edge_attr = self.edge_projector(edge_attr.unsqueeze(-1))
        attn_out = self.linear(self.attention(x, edge_index, edge_attr=projected_edge_attr))
        x = self.norm1(x + attn_out)
        return self.norm2(x + self.ffn(x))

def make_batched_edge_index(edge_index, num_genes, batch_size, device):
    rows, cols = [], []
    for b in range(batch_size):
        offset = b * num_genes
        rows.append(edge_index[0] + offset)
        cols.append(edge_index[1] + offset)
    return torch.stack([torch.cat(rows), torch.cat(cols)], dim=0).to(device)

def make_batched_edge_attr(edge_attr, batch_size, device):
    return edge_attr.repeat(batch_size).to(device)

class HybridGraphTransformer(nn.Module):
    def __init__(self, num_local_layers, num_global_layers, embed_dim, num_heads, dropout, max_batch_size=32):
        super().__init__()
        self.local_encoder = nn.ModuleList([GraphTransformerLayer(embed_dim, num_heads, dropout) for _ in range(num_local_layers)])
        global_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 2, dropout=dropout, batch_first=True)
        self.global_encoder = nn.TransformerEncoder(global_layer, num_layers=num_global_layers)
        self.final_norm = nn.LayerNorm(embed_dim)
        self._batched_edge_cache = {}
        self._max_batch_size = max_batch_size

    def forward(self, gene_feats, cls_tokens, edge_index, edge_attr, batch_size):
        B, G, D = gene_feats.shape
        x_flat = gene_feats.reshape(B * G, D)
        cache_key = (batch_size, edge_index.device)
        if cache_key not in self._batched_edge_cache:
            batched_edge_index = make_batched_edge_index(edge_index, G, B, device=gene_feats.device)
            batched_edge_attr = make_batched_edge_attr(edge_attr, B, device=gene_feats.device)
            if batch_size <= self._max_batch_size: self._batched_edge_cache[cache_key] = (batched_edge_index, batched_edge_attr)
        else:
            batched_edge_index, batched_edge_attr = self._batched_edge_cache[cache_key]

        for layer in self.local_encoder: x_flat = layer(x_flat, batched_edge_index, batched_edge_attr)

        x_local_genes = x_flat.reshape(B, G, D)
        full_sequence = torch.cat([cls_tokens, x_local_genes], dim=1)

        if self.training:
            for layer in self.global_encoder.layers:
                full_sequence = torch.utils.checkpoint.checkpoint(layer, full_sequence, use_reentrant=False)
        else:
            full_sequence = self.global_encoder(full_sequence)

        return self.final_norm(full_sequence)

class FoundationalGeneTransformer(pl.LightningModule):
    def __init__(self, config, num_genes, go_feature_dim, pos_encoding_dim, edge_index, edge_attr, go_features, pos_enc, full_vocab_size, pad_token_id, **kwargs):
        super().__init__()
        # --- CORRECTED HYPERPARAMETER SAVING ---
        self.save_hyperparameters(ignore=['edge_index', 'edge_attr', 'go_features', 'pos_enc'])

        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_attr', edge_attr)
        self.register_buffer('go_features', go_features)
        self.register_buffer('pos_enc', pos_enc)
        self.register_buffer('gene_ids', torch.arange(num_genes))

        self.embedding_layer = GeneExpressionEmbedding(
            num_genes, go_feature_dim, pos_encoding_dim, self.hparams.config['embedding_dim'], full_vocab_size, pad_token_id
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hparams.config['embedding_dim']))
        self.encoder = HybridGraphTransformer(
            self.hparams.config['num_local_layers'], self.hparams.config['num_global_layers'], self.hparams.config['embedding_dim'], self.hparams.config['num_heads'], self.hparams.config['dropout']
        )

    def forward(self, x_values):
        batch_size = x_values.size(0)
        full_embeddings = self.embedding_layer(self.gene_ids, x_values, self.go_features, self.pos_enc)
        cls_token_expanded = self.cls_token.expand(batch_size, -1, -1)
        encoded_features = self.encoder(
            full_embeddings, cls_token_expanded, self.edge_index, self.edge_attr, batch_size
        )
        return None, encoded_features[:, 0]

def load_foundation_from_legacy_checkpoint(config, num_genes, go_feature_dim, pos_encoding_dim, edge_index, edge_attr, go_features, pos_enc, full_vocab_size, pad_token_id, checkpoint_path, map_location='cpu'):
    print(f"--- Manually loading and adapting legacy checkpoint: {checkpoint_path} ---")
    # --- ADDED weights_only=True FOR SECURITY ---
    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    state_dict = ckpt['state_dict']
    old_num_genes = state_dict['embedding_layer.gene_embedding.weight'].shape[0]
    old_go_dim = state_dict['go_features'].shape[1]
    old_pos_dim = state_dict['pos_enc'].shape[1]
    model = FoundationalGeneTransformer(
        config=config, num_genes=old_num_genes, go_feature_dim=old_go_dim, pos_encoding_dim=old_pos_dim,
        edge_index=state_dict['edge_index'], edge_attr=state_dict['edge_attr'], go_features=state_dict['go_features'],
        pos_enc=state_dict['pos_enc'], full_vocab_size=full_vocab_size, pad_token_id=pad_token_id
    )
    model.load_state_dict(state_dict, strict=False)
    print(f"Adapting model from {old_num_genes} genes to {num_genes} genes...")
    model.embedding_layer.gene_embedding = nn.Embedding(num_genes, config['embedding_dim'])
    model.register_buffer('edge_index', edge_index); model.register_buffer('edge_attr', edge_attr)
    model.register_buffer('go_features', go_features); model.register_buffer('pos_enc', pos_enc)
    model.register_buffer('gene_ids', torch.arange(num_genes))
    model.hparams.num_genes, model.hparams.go_feature_dim, model.hparams.pos_encoding_dim = num_genes, go_feature_dim, pos_encoding_dim
    print("Model adaptation complete.")
    return model

# =====================================================================================
# BATCH CORRECTION TASK-SPECIFIC CLASSES
# =====================================================================================
class BatchCorrDataset(Dataset):
    def __init__(self, adata, indices, bins):
        self.indices = indices
        X = adata.X[indices]
        if hasattr(X, 'toarray'): X = X.toarray()
        self.expr_float = X.astype(np.float32)
        self.binned = np.digitize(self.expr_float, bins=bins, right=True).astype(np.int64) if bins is not None else np.floor(self.expr_float * 511).astype(np.int64)
        batches = adata.obs[BATCH_KEY].astype('category')
        self.batch_codes = batches.cat.codes.values[indices].astype(np.int64)
        self.num_batches = len(batches.cat.categories)

    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        return (torch.from_numpy(self.binned[idx]).long(),
                torch.from_numpy(self.expr_float[idx]),
                torch.tensor(self.batch_codes[idx], dtype=torch.long))

class BatchCorrDataModule(pl.LightningDataModule):
    def __init__(self, adata, bins, train_idx, val_idx, batch_size=32):
        super().__init__()
        self.adata = adata
        self.bins = bins
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.batch_size = batch_size
    def setup(self, stage=None):
        self.train_ds = BatchCorrDataset(self.adata, self.train_idx, self.bins)
        self.val_ds = BatchCorrDataset(self.adata, self.val_idx, self.bins)
    def train_dataloader(self): return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4)
    def val_dataloader(self): return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=4)

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None
def grad_reverse(x, lambd): return GradientReversal.apply(x, lambd)

class BatchAdversarialFinetuner(pl.LightningModule):
    def __init__(self, backbone: FoundationalGeneTransformer, num_batches: int, num_genes_to_reconstruct: int, model_gene_indices: torch.Tensor, lr: float, lambda_adv: float, freeze_backbone: bool):
        super().__init__()
        self.save_hyperparameters(ignore=['backbone', 'model_gene_indices'])
        self.backbone = backbone
        self.register_buffer('model_gene_indices', model_gene_indices)
        emb_dim = self.backbone.hparams.config['embedding_dim']
        self.recon_head = nn.Linear(emb_dim, num_genes_to_reconstruct)
        self.batch_discriminator = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(emb_dim // 2, num_batches)
        )
        if freeze_backbone: self.backbone.freeze()
        self.recon_loss_fn = nn.MSELoss()
        self.adv_loss_fn = nn.CrossEntropyLoss()
        self.train_batch_acc = Accuracy(task='multiclass', num_classes=num_batches)
        self.val_batch_acc = Accuracy(task='multiclass', num_classes=num_batches)

    def forward(self, x_subset):
        batch_size = x_subset.size(0)
        num_total_genes = self.backbone.hparams.num_genes
        pad_token_id = self.backbone.hparams.pad_token_id
        full_input = torch.full((batch_size, num_total_genes), fill_value=pad_token_id, device=x_subset.device, dtype=x_subset.dtype)
        full_input[:, self.model_gene_indices] = x_subset
        with torch.set_grad_enabled(not self.hparams.freeze_backbone):
            _, cell_repr = self.backbone(full_input)
        return cell_repr, self.recon_head(cell_repr)

    def training_step(self, batch, batch_idx):
        binned_x, float_x, batch_labels = batch
        cell_repr, recon_expr = self.forward(binned_x)
        recon_loss = self.recon_loss_fn(recon_expr, float_x)
        reversed_repr = grad_reverse(cell_repr, self.hparams.lambda_adv)
        logits_batch = self.batch_discriminator(reversed_repr)
        adv_loss = self.adv_loss_fn(logits_batch, batch_labels)
        total_loss = recon_loss + adv_loss
        self.log_dict({'train_recon_loss': recon_loss, 'train_adv_loss': adv_loss, 'train_batch_acc': self.train_batch_acc(logits_batch, batch_labels)}, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        binned_x, float_x, batch_labels = batch
        cell_repr, recon_expr = self.forward(binned_x)
        recon_loss = self.recon_loss_fn(recon_expr, float_x)
        logits_batch = self.batch_discriminator(cell_repr)
        adv_loss = self.adv_loss_fn(logits_batch, batch_labels)
        self.log_dict({'val_recon_loss': recon_loss, 'val_adv_loss': adv_loss, 'val_batch_acc': self.val_batch_acc(logits_batch, batch_labels)}, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.lr)

# =====================================================================================
# MAIN EXECUTION SCRIPT
# =====================================================================================
def main():
    # --- Step 1: Load ORIGINAL graph components and full model as a weight source ---
    print("--- Loading ORIGINAL graph components ---")
    graph_gene_names = np.load(GENE_NAMES_PATH, allow_pickle=True)
    num_original_genes = len(graph_gene_names)
    go_features_full = torch.from_numpy(np.load(GO_FEATURES_PATH)).float()
    adj_full = sp.load_npz(GRAPH_PATH)
    pos_enc_full = torch.from_numpy(np.load(POS_ENC_CACHE)).float()
    config = {'embedding_dim': 128, 'hidden_channels': 256, 'num_heads': 2, 'num_local_layers': 2, 'num_global_layers': 4, 'dropout': 0.1}
    bins = np.load(EXPR_BINS_PATH) if os.path.exists(EXPR_BINS_PATH) else None
    vocab_size = (len(bins) + 1) if bins is not None else 512
    pad_token_id = vocab_size

    full_backbone_weight_source = load_foundation_from_legacy_checkpoint(
        config=config, num_genes=num_original_genes, go_feature_dim=go_features_full.shape[1], pos_encoding_dim=pos_enc_full.shape[1],
        edge_index=from_scipy_sparse_matrix(adj_full)[0], edge_attr=from_scipy_sparse_matrix(adj_full)[1].float(),
        go_features=go_features_full, pos_enc=pos_enc_full, full_vocab_size=vocab_size + 2, pad_token_id=pad_token_id,
        checkpoint_path=PRETRAINED_CKPT
    )
    full_backbone_weight_source.eval()

    # --- Step 2: Load AnnData and Define Subgraph ---
    combined_adata = load_and_combine_adata(REFERENCE_ADATA_PATH, QUERY_ADATA_PATH, MS_GENE_COL)

    print("--- Finding initial gene overlap ---")
    common_genes = sorted(list(set(graph_gene_names) & set(combined_adata.var_names)))
    if not common_genes: raise RuntimeError("No common genes found.")
    num_common_genes = len(common_genes)
    graph_gene_to_idx = {name: i for i, name in enumerate(graph_gene_names)}
    seed_node_indices = np.array([graph_gene_to_idx[name] for name in common_genes])
    print(f"Found {num_common_genes} seed genes.")

    print("--- Expanding gene set with 2-hop neighbors ---")
    adj_binary = (adj_full > 0).astype(int).tocsr()
    adj_2_hop = adj_binary @ adj_binary
    one_hop_neighbors = adj_binary[seed_node_indices].sum(axis=0).nonzero()[1]
    two_hop_neighbors = adj_2_hop[seed_node_indices].sum(axis=0).nonzero()[1]
    graph_sel_idx = np.union1d(np.union1d(seed_node_indices, one_hop_neighbors), two_hop_neighbors)
    print(f"Expanded gene set to {len(graph_sel_idx)} nodes.")

    # --- Step 3: Create and Prepare Subgraph Model for Finetuning ---
    num_subgraph_genes = len(graph_sel_idx)
    go_features_sub = go_features_full[graph_sel_idx]
    pos_enc_sub = pos_enc_full[graph_sel_idx]
    adj_sub = adj_full.tocsr()[graph_sel_idx, :][:, graph_sel_idx]
    edge_index_sub, edge_attr_sub = from_scipy_sparse_matrix(adj_sub)

    print(f"--- Creating a new backbone for the {num_subgraph_genes}-gene subgraph ---")
    finetune_backbone = FoundationalGeneTransformer(
        config=config, num_genes=num_subgraph_genes, go_feature_dim=go_features_sub.shape[1], pos_encoding_dim=pos_enc_sub.shape[1],
        edge_index=edge_index_sub, edge_attr=edge_attr_sub.float(), go_features=go_features_sub, pos_enc=pos_enc_sub,
        full_vocab_size=vocab_size + 2, pad_token_id=pad_token_id
    )

    print("--- Surgically copying weights from full model to subgraph model ---")
    source_state_dict = full_backbone_weight_source.state_dict()
    target_state_dict = finetune_backbone.state_dict()
    new_state_dict = {}
    for name, param in target_state_dict.items():
        if name in source_state_dict and source_state_dict[name].shape == param.shape:
            new_state_dict[name] = source_state_dict[name].clone()
    new_state_dict['embedding_layer.gene_embedding.weight'] = source_state_dict['embedding_layer.gene_embedding.weight'][graph_sel_idx].clone()
    finetune_backbone.load_state_dict(new_state_dict, strict=False)
    print("Weights copied successfully.")

    adata_filtered = combined_adata[:, common_genes].copy()

    # --- Step 4: Stratified Data Splitting ---
    print("\n--- Performing stratified split on the combined dataset ---")
    all_indices = np.arange(adata_filtered.n_obs)
    # Use MS_LABEL_COL for stratification to maintain cell type distribution
    labels = adata_filtered.obs[MS_LABEL_COL].astype('category').cat.codes.values

    # First split: 85% for training+validation, 15% for testing
    train_val_indices, test_indices, train_val_labels, _ = train_test_split(
        all_indices, labels, test_size=0.15, random_state=42, stratify=labels
    )
    # Second split: Split the 85% into train and validation sets (70% train, 15% val of total)
    train_indices, val_indices, _, _ = train_test_split(
        train_val_indices, train_val_labels, test_size=(0.15 / 0.85), random_state=42, stratify=train_val_labels
    )
    print(f"Data split: {len(train_indices)} train, {len(val_indices)} validation, {len(test_indices)} test cells.\n")


    # --- Step 5: Setup Finetuner and Trainer ---
    dm = BatchCorrDataModule(adata_filtered, bins, train_idx=train_indices, val_idx=val_indices, batch_size=BATCH_SIZE)
    dm.setup()

    subgraph_orig_idx_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(graph_sel_idx)}
    model_gene_indices = torch.tensor([subgraph_orig_idx_map[i] for i in seed_node_indices], dtype=torch.long)

    model = BatchAdversarialFinetuner(
        backbone=finetune_backbone, num_batches=dm.train_ds.num_batches,
        num_genes_to_reconstruct=num_common_genes, model_gene_indices=model_gene_indices,
        lr=LR, lambda_adv=LAMBDA_ADV, freeze_backbone=FREEZE_BACKBONE
    )

    trainer = pl.Trainer(
        accelerator='gpu', devices=1, max_epochs=MAX_EPOCHS,
        precision='16-mixed',
        callbacks=[
            ModelCheckpoint(dirpath=OUTPUT_DIR, monitor='val_recon_loss', mode='min', filename='batchcorr-ms-subgraph-{epoch:02d}-{val_recon_loss:.3f}'),
            EarlyStopping(monitor='val_recon_loss', mode='min', patience=10)
        ],
        logger=WandbLogger(project='scgt-finetune', name='batchcorr-ms-subgraph')
    )

    trainer.fit(model, dm)

    print(f'Finished adversarial batch correction. Best model saved in {OUTPUT_DIR}')

    print("\n--- Evaluating on the stratified test set ---")
    test_ds = BatchCorrDataset(adata_filtered, test_indices, bins)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=4)
    # Note: trainer.test is not implemented for this model, but we run it to show how.
    # It will log 'val_recon_loss' and 'val_batch_acc' for the test set.
    trainer.test(model, dataloaders=test_loader, ckpt_path='best')


if __name__ == '__main__':
    main()