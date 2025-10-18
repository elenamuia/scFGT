# train_gene_scgt.py (with weighted edge support)

import os
# Hardcoded GPU list (always use physical GPUs 1,2,3)
GPU_DEVICE_LIST = "1,2,3"
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_DEVICE_LIST

# GPU memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

os.environ["NCCL_P2P_DISABLE"] = "1"
# NCCL robustness flags
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_TIMEOUT"] = str(60 * 60 * 6)  # 6 hours in seconds

import re
import glob
import gc
import functools
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import from_scipy_sparse_matrix, k_hop_subgraph
from pytorch_lightning.loggers import WandbLogger
import anndata as an
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, EarlyStopping
import scanpy as sc
from pytorch_lightning.strategies import DDPStrategy
from datetime import timedelta
import torchmetrics
import hashlib

def dist_rank():
    if dist.is_available() and dist.is_initialized():
        try:
            return dist.get_rank()
        except Exception:
            return 0
    return 0

# Enable Tensor Core friendly float32 matmul (performance boost on Ampere+)
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

def should_init_dist():
    # Honor override
    force = os.getenv('FORCE_DDP', '0').lower() in ('1','true','yes')
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    has_rank = 'RANK' in os.environ and 'LOCAL_RANK' in os.environ
    # Slurm support (common cluster env vars)
    slurm_world = 'SLURM_NTASKS' in os.environ and int(os.getenv('SLURM_NTASKS','1')) > 1
    if force:
        return world_size > 1
    return (world_size > 1 and has_rank) or slurm_world

# Custom callback for monitoring training progress and GPU usage
class GPUMonitorCallback(Callback):
    def __init__(self, log_every_n_epochs=5):
        self.log_every_n_epochs = log_every_n_epochs
    
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.log_every_n_epochs == 0:
            if trainer.is_global_zero:
                pass

# ----------------------------- CONFIG (LIGHTER) ---------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_vocab_size(bins_path: str, fallback: int = 512):
    if bins_path and os.path.exists(bins_path):
        try:
            bins = np.load(bins_path)
            # bins defines bin edges; vocab = number_of_bins + 1
            vs = int(len(bins) + 1)
            if vs < 16:  # sanity guard
                return fallback
            return vs
        except Exception:
            return fallback
    return fallback

EXPR_BINS_PATH = 'outputs/expression_bins.npy'
EXPRESSION_VOCAB_SIZE = get_vocab_size(EXPR_BINS_PATH)
PAD_TOKEN_ID = EXPRESSION_VOCAB_SIZE
MASK_TOKEN_ID = EXPRESSION_VOCAB_SIZE + 1
FULL_VOCAB_SIZE = EXPRESSION_VOCAB_SIZE + 2

CONFIG = {
    "embedding_dim": 128,            # reduced from 256
    "hidden_channels": 256,          # reduced from 512
    "num_heads": 2,                  # reduced from 4
    "num_local_layers": 2,           # reduced from 2
    "num_global_layers": 4,          # reduced from 3
    "dropout": 0.1,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,            # lowered
    "warmup_steps": 500,
    # Hybrid loss weighting: alpha * NB + (1-alpha) * MRE
    "alpha_nb": 0.7,
    # Contrastive (kept but disabled by default)
    "use_contrastive_loss": False,
    "contrastive_projection_dim": 64,
    "contrastive_temperature": 0.1,
    "contrastive_weight": 0.1,
}

DATA_MODULE_CONFIG = {
    "batch_size": 32,   # increased from 8 due to lighter model
    "mask_prob": 0.10,
    "num_workers": 4,
}

TRAINER_CONFIG = {
    "max_epochs": 120,  # slightly reduced for quicker cycles
    "log_every_n_steps": 5,
    "precision": "16-mixed",
    "accumulate_grad_batches": 2,
    "sync_batchnorm": True,
    "gradient_clip_val": 0.5
}

RAW_ADATA_PATH = 'data/PBMCs/ycpu.h5ad'
ZARR_PATH = RAW_ADATA_PATH.replace('.h5ad', '.zarr')
INPUT_GRAPH_PATH = 'outputs/A_context_multimodal_4k.npz'
GO_FEATURES_PATH = 'outputs/gene_go_features_4k.npy'
GENE_NAMES_PATH = 'outputs/graph_gene_names_4k.npy'
POS_ENC_CACHE = 'outputs/laplacian_pos_enc_32.npy'
CHECKPOINT_DIR = 'checkpoints/check_w_feat'
OUTPUT_DIR = 'cell_embeddings/'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------- Utilities ---------------------------------

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def print_gpu_memory(prefix=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        if not dist.is_available() or not dist.is_initialized() or dist_rank() == 0:
            print(f"{prefix} GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")

def print_all_gpu_status():
    if torch.cuda.is_available() and (not dist.is_available() or not dist.is_initialized() or dist_rank() == 0):
        print("="*60)
        print("GPU Status:")
        print("="*60)
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            if i == torch.cuda.current_device():
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                cached = torch.cuda.memory_reserved(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached, {total:.2f}GB total")
            else:
                props = torch.cuda.get_device_properties(i)
                print(f"  Total Memory: {props.total_memory / 1024**3:.2f}GB")
        print("="*60)

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

def make_batched_edge_attr(edge_attr, batch_size, device=None): # <<< NEW
    """Repeats edge attributes for each item in the batch."""
    batched = edge_attr.repeat(batch_size)
    if device is not None:
        batched = batched.to(device)
    return batched

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

# ------------------------- Model components ---------------------------

class GeneExpressionEmbedding(nn.Module):
    def __init__(self, num_genes, go_feature_dim, pos_encoding_dim, embedding_dim):
        super().__init__()
        self.gene_embedding = nn.Embedding(num_genes, embedding_dim)
        self.value_embedding = nn.Embedding(FULL_VOCAB_SIZE, embedding_dim, padding_idx=PAD_TOKEN_ID)
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
        # <<< MODIFIED: GATv2Conv setup to handle edge features
        edge_feature_dim = 16  # Internal dimension for projected edge weights
        self.attention = GATv2Conv(
            in_channels=embed_dim,
            out_channels=embed_dim,
            heads=num_heads,
            concat=True,
            dropout=dropout,
            edge_dim=edge_feature_dim # Tell GATv2 to expect edge features
        )
        self.edge_projector = nn.Linear(1, edge_feature_dim) # Project 1D weight to edge_feature_dim
        # <<< END MODIFIED
        
        self.linear = nn.Linear(embed_dim * num_heads, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, edge_index, edge_attr): # <<< MODIFIED: Accept edge_attr
        # x: [N, D], edge_attr: [num_edges, 1]
        
        # <<< NEW: Project edge weights and pass to GATv2Conv
        projected_edge_attr = self.edge_projector(edge_attr.unsqueeze(-1))
        attn_out = self.linear(self.attention(x, edge_index, edge_attr=projected_edge_attr))
        # <<< END NEW
        
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

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

    def forward(self, gene_feats, cls_tokens, edge_index, edge_attr, batch_size): # <<< MODIFIED: Accept edge_attr
        B, G, D = gene_feats.shape
        assert B == batch_size, 'Batch size mismatch'

        x_flat = gene_feats.reshape(B * G, D)

        # Cache key now includes device to be safer
        cache_key = (batch_size, edge_index.device)

        if cache_key not in self._batched_edge_cache:
            # <<< MODIFIED: Batch edge_index and edge_attr together
            batched_edge_index = make_batched_edge_index(edge_index, G, batch_size, device=edge_index.device)
            batched_edge_attr = make_batched_edge_attr(edge_attr, batch_size, device=edge_attr.device)
            if batch_size <= self._max_batch_size:
                self._batched_edge_cache[cache_key] = (batched_edge_index, batched_edge_attr)
        else:
            batched_edge_index, batched_edge_attr = self._batched_edge_cache[cache_key]
            # <<< END MODIFIED

        x_local_flat = x_flat
        for layer in self.local_encoder:
            # <<< MODIFIED: Pass batched edge attributes to the layer
            x_local_flat = layer(x_local_flat, batched_edge_index, batched_edge_attr)
            # <<< END MODIFIED
        
        x_local_genes = x_local_flat.reshape(B, G, D)

        full_sequence_input = torch.cat([cls_tokens, x_local_genes], dim=1)
        processed_sequence = full_sequence_input
        for layer in self.global_encoder.layers:
             processed_sequence = torch.utils.checkpoint.checkpoint(
                layer, processed_sequence, use_reentrant=False
            )
        out = self.final_norm(processed_sequence)
        return out

# removed obsolete MGMPredictionHead

class NBPredictionHead(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Linear(embed_dim, 2)
    def forward(self, x):
        out = self.net(x)
        return out[..., 0], out[..., 1]

def neg_binom_loss(counts, log_mu, log_r, mask=None, eps=1e-8):
    mu, r, x = torch.exp(log_mu).clamp(min=eps), torch.exp(log_r).clamp(min=eps), counts.float()
    ll = (torch.lgamma(x + r) - torch.lgamma(r) - torch.lgamma(x + 1.0) +
          r * (torch.log(r) - torch.log(mu + r)) + x * (torch.log(mu) - torch.log(mu + r)))
    nll = -ll
    if mask is not None:
        mask_f = mask.bool()
        if mask_f.any():
            return nll[mask_f].mean()
        return nll.new_tensor(0.0)
    return nll.mean()

def masked_relative_error(counts, pred_mu, mask, eps=1.0):
    # eps=1.0 avoids exploding relative error for zero counts
    if mask is None or not mask.any():
        return pred_mu.new_tensor(0.0)
    m = mask.bool()
    denom = counts[m].float() + eps
    rel = (pred_mu[m] - counts[m].float()).abs() / denom
    return rel.mean()

# removed focal_loss and compute_class_weights (unused)

def create_scheduler_with_warmup(optimizer, warmup_steps, total_steps, eta_min=1e-6):
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=total_steps - warmup_steps, eta_min=eta_min)
    return SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

# ... All Dataset and DataModule classes remain unchanged ...
# (FullyPrematerializedDataset, PrematerializedSingleCellDataset, ContrastiveStructuralMaskDataset, SingleCellDataModule)
# NOTE: The provided file content for these classes is assumed to be correct and is included here as is.

class FullyPrematerializedDataset(Dataset):
    def __init__(self, cache_dir, cache_key, split_name):
        super().__init__()
        self.cache_dir, self.cache_key, self.split_name = Path(cache_dir), cache_key, split_name
        if dist_rank() == 0:
            print(f"Loading {split_name} dataset from cache (key: {cache_key})")
        self.masked_inputs = torch.load(self.cache_dir / f'{split_name}_masked_input_{cache_key}.pt')
        self.masks = torch.load(self.cache_dir / f'{split_name}_masks_{cache_key}.pt')
        self.labels = torch.load(self.cache_dir / f'{split_name}_labels_{cache_key}.pt')
        self.counts = torch.load(self.cache_dir / f'{split_name}_counts_{cache_key}.pt')
        if dist_rank() == 0:
            print(f"  {split_name} dataset loaded: {len(self.masked_inputs)} samples")
    def __len__(self): return len(self.masked_inputs)
    def __getitem__(self, idx): return (self.masked_inputs[idx], self.masks[idx], self.labels[idx], self.counts[idx])

class PrematerializedSingleCellDataset(Dataset):
    def __init__(self, adata_path, indices, mask_prob, bins_path=None, zarr_path=None, var_idx=None, cache_dir=None):
        super().__init__()
        self.zarr_path = zarr_path or adata_path.replace('.h5ad', '.zarr')
        self.indices, self.mask_prob = np.asarray(indices, dtype=np.int64), mask_prob
        self.var_idx = None if var_idx is None else np.asarray(var_idx, dtype=np.int64)
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(self.zarr_path), 'prematerialized_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        config_str = f"{self.zarr_path}_{len(indices)}_{var_idx is not None}_{mask_prob}"
        self.cache_key = hashlib.md5(config_str.encode()).hexdigest()[:8]
        self.counts_cache_path, self.binned_cache_path = os.path.join(self.cache_dir, f'counts_{self.cache_key}.npy'), os.path.join(self.cache_dir, f'binned_{self.cache_key}.npy')
        self.bins = np.load(bins_path) if bins_path and os.path.exists(bins_path) else None
        self._prematerialize_data()
    def _prematerialize_data(self):
        if os.path.exists(self.counts_cache_path) and os.path.exists(self.binned_cache_path):
            if dist_rank() == 0: print(f"Loading prematerialized data from cache (key: {self.cache_key})")
            self.counts_data, self.binned_data = np.load(self.counts_cache_path), np.load(self.binned_cache_path)
            return
        if dist_rank() == 0: print(f"Prematerializing dataset for {len(self.indices)} samples...")
        adata = an.read_zarr(self.zarr_path)
        expr_matrix = adata[self.indices, self.var_idx].X if self.var_idx is not None else adata[self.indices].X
        expr_matrix = expr_matrix.toarray() if hasattr(expr_matrix, 'toarray') else np.asarray(expr_matrix, dtype=np.float32)
        self.counts_data = expr_matrix.astype(np.int32)
        self.binned_data = np.digitize(expr_matrix, bins=self.bins, right=True).astype(np.int64) if self.bins is not None else np.floor((expr_matrix / np.maximum(expr_matrix.max(axis=0), 1.0)) * (EXPRESSION_VOCAB_SIZE - 1)).astype(np.int64)
        if dist_rank() == 0:
            print(f"Saving prematerialized data to cache (key: {self.cache_key})")
        np.save(self.counts_cache_path, self.counts_data)
        np.save(self.binned_cache_path, self.binned_data)
        del adata, expr_matrix; gc.collect()
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        d_vals, counts = torch.from_numpy(self.binned_data[idx]).long(), torch.from_numpy(self.counts_data[idx]).long()
        is_expressed = d_vals > 0
        mask = (torch.rand(len(d_vals)) < self.mask_prob) & is_expressed
        labels, masked_input = d_vals.clone(), d_vals.clone()
        masked_input[mask] = MASK_TOKEN_ID
        return masked_input, mask, labels, counts

def worker_init_fn(worker_id): np.random.seed(torch.initial_seed() % (2**32))

class ContrastiveStructuralMaskDataset(Dataset):
    def __init__(self, adata_path, indices, mask_prob, edge_index, num_genes, k_hops=1, bins_path=None, zarr_path=None, var_idx=None):
        super().__init__()
        self.zarr_path, self.indices, self.mask_prob, self.var_idx, self.num_genes = zarr_path or adata_path.replace('.h5ad', '.zarr'), np.asarray(indices, dtype=np.int64), mask_prob, None if var_idx is None else np.asarray(var_idx, dtype=np.int64), num_genes
        cache_dir = os.path.join(os.path.dirname(self.zarr_path), 'prematerialized_cache')
        os.makedirs(cache_dir, exist_ok=True)
        config_str, cache_key = f"{self.zarr_path}_{len(self.indices)}_{self.var_idx is not None}", hashlib.md5(config_str.encode()).hexdigest()[:8]
        self.counts_cache_path, self.binned_cache_path = os.path.join(cache_dir, f'counts_{cache_key}.npy'), os.path.join(cache_dir, f'binned_{cache_key}.npy')
        self.bins = np.load(bins_path) if bins_path and os.path.exists(bins_path) else None
        self._prematerialize_data()
        assert self.binned_data.shape[1] == self.num_genes, "Gene mismatch!"
        if dist_rank() == 0: print(f"Pre-computing {k_hops}-hop neighborhoods...")
        self.gene_neighborhoods = [k_hop_subgraph(i, k_hops, edge_index.cpu(), False)[0] for i in tqdm(range(self.num_genes), disable=dist_rank() != 0)]
    def _prematerialize_data(self):
        if os.path.exists(self.counts_cache_path) and os.path.exists(self.binned_cache_path):
            if dist_rank() == 0: print("Loading prematerialized data...")
            self.counts_data, self.binned_data = np.load(self.counts_cache_path), np.load(self.binned_cache_path)
            return
        if dist_rank() == 0: print(f"Prematerializing dataset...")
        adata = an.read_zarr(self.zarr_path)
        expr_matrix = adata[self.indices, self.var_idx].X if self.var_idx is not None else adata[self.indices].X
        expr_matrix = expr_matrix.toarray() if hasattr(expr_matrix, 'toarray') else np.asarray(expr_matrix, dtype=np.float32)
        self.counts_data, self.binned_data = expr_matrix.astype(np.int32), np.digitize(expr_matrix, bins=self.bins, right=True).astype(np.int64) if self.bins is not None else np.floor((expr_matrix / np.maximum(expr_matrix.max(axis=0), 1.0)) * (EXPRESSION_VOCAB_SIZE - 1)).astype(np.int64)
        if dist_rank() == 0:
            print("Saving prematerialized data...")
        np.save(self.counts_cache_path, self.counts_data)
        np.save(self.binned_cache_path, self.binned_data)
        del adata, expr_matrix; gc.collect()
    def __len__(self): return len(self.indices)
    def _create_masked_view_structural(self, d_vals_tensor):
        expressed_indices = (d_vals_tensor > 0).nonzero(as_tuple=True)[0]
        if len(expressed_indices) == 0: return d_vals_tensor.clone(), torch.zeros_like(d_vals_tensor, dtype=torch.bool), d_vals_tensor.clone()
        num_to_mask_total, genes_to_mask_set, potential_centers = int(len(expressed_indices) * self.mask_prob), set(), expressed_indices.tolist()
        while len(genes_to_mask_set) < num_to_mask_total and potential_centers:
            center_idx = potential_centers.pop(torch.randint(0, len(potential_centers), (1,)).item())
            genes_to_mask_set.update(self.gene_neighborhoods[center_idx].tolist())
        final_mask = torch.zeros_like(d_vals_tensor, dtype=torch.bool)
        if genes_to_mask_set:
            final_mask[torch.tensor(list(genes_to_mask_set), dtype=torch.long)] = True
            final_mask &= (d_vals_tensor > 0)
        labels, masked_input = d_vals_tensor.clone(), d_vals_tensor.clone()
        masked_input[final_mask] = MASK_TOKEN_ID
        return masked_input, final_mask, labels
    def __getitem__(self, idx):
        d_vals, counts = torch.from_numpy(self.binned_data[idx]).long(), torch.from_numpy(self.counts_data[idx]).long()
        view1 = self._create_masked_view_structural(d_vals)
        view2 = self._create_masked_view_structural(d_vals)
        return (view1[0], view1[1], view1[2], counts), (view2[0], view2[1], view2[2], counts.clone())

class SingleCellDataModule(pl.LightningDataModule):
    def __init__(self, adata_path, batch_size, mask_prob, num_workers, edge_index=None, num_genes=None, use_structural_masking=False, use_contrastive_loss=False, use_prematerialized=True, bins_cache='outputs/expression_bins.npy', gene_names_path=None):
        super().__init__()
        self.save_hyperparameters()
        self.zarr_path, self.compact_zarr_path = adata_path.replace('.h5ad', '.zarr'), adata_path.replace('.h5ad', '.graph.zarr')
        self.var_idx = None
    def setup(self, stage=None):
        use_zarr = self.compact_zarr_path if os.path.exists(self.compact_zarr_path) else self.zarr_path
        adata = an.read_zarr(use_zarr)
        indices, sp_pt = np.arange(adata.n_obs), int(adata.n_obs * 0.9)
        np.random.shuffle(indices)
        self.train_indices, self.val_indices = indices[:sp_pt], indices[sp_pt:]
        dataset_args = {'zarr_path': use_zarr, 'var_idx': self.var_idx}
        DatasetClass = ContrastiveStructuralMaskDataset if self.hparams.use_contrastive_loss else PrematerializedSingleCellDataset
        if self.hparams.use_contrastive_loss: dataset_args.update({'edge_index': self.hparams.edge_index, 'num_genes': self.hparams.num_genes})
        self.train_dataset = DatasetClass(self.hparams.adata_path, self.train_indices, self.hparams.mask_prob, bins_path=self.hparams.bins_cache, **dataset_args)
        self.val_dataset = DatasetClass(self.hparams.adata_path, self.val_indices, self.hparams.mask_prob, bins_path=self.hparams.bins_cache, **dataset_args)
    def _get_loader(self, dataset):
        is_fully_prematerialized, is_prematerialized = isinstance(dataset, FullyPrematerializedDataset), self.hparams.use_prematerialized
        num_workers = min(12, self.hparams.num_workers*3) if is_fully_prematerialized else min(8, self.hparams.num_workers*2) if is_prematerialized else self.hparams.num_workers
        prefetch = 16 if is_fully_prematerialized else 8 if is_prematerialized else 4
        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, worker_init_fn=worker_init_fn, persistent_workers=True, prefetch_factor=prefetch, drop_last=True)
    def train_dataloader(self): return self._get_loader(self.train_dataset)
    def val_dataloader(self): return self._get_loader(self.val_dataset)


# -------------------- Lightning Module (model) -------------------------

class FoundationalGeneTransformer(pl.LightningModule):
    def __init__(self, hparams, num_genes, go_feature_dim, pos_encoding_dim, edge_index, edge_attr, go_features, pos_enc):
        super().__init__()
        self.save_hyperparameters(hparams)
        cfg = self.hparams['config']
        self.alpha_nb = cfg.get('alpha_nb', 0.5)

        # Buffers
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_attr', edge_attr)
        self.register_buffer('go_features', go_features)
        self.register_buffer('pos_enc', pos_enc)
        self.register_buffer('gene_ids', torch.arange(num_genes, dtype=torch.long))

        # Embedding & Encoder
        self.embedding_layer = GeneExpressionEmbedding(
            num_genes=num_genes, go_feature_dim=go_feature_dim,
            pos_encoding_dim=pos_encoding_dim, embedding_dim=cfg['embedding_dim']
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg['embedding_dim']))
        self.encoder = HybridGraphTransformer(
            num_local_layers=cfg['num_local_layers'], num_global_layers=cfg['num_global_layers'],
            embed_dim=cfg['embedding_dim'], num_heads=cfg['num_heads'], dropout=cfg['dropout'],
            max_batch_size=max(32, self.hparams.data_config.get('batch_size', 8))
        )
        self.nb_head = NBPredictionHead(cfg['embedding_dim'])

        # Optional contrastive
        if cfg.get('use_contrastive_loss', False):
            proj_dim = cfg.get('contrastive_projection_dim', 64)
            self.contrastive_head = nn.Sequential(
                nn.Linear(cfg['embedding_dim'], cfg['embedding_dim']), nn.ReLU(),
                nn.Linear(cfg['embedding_dim'], proj_dim)
            )

        # Metrics (simplified)
        for stage in ['train', 'val']:
            for metric_name, metric_cls in [
                ('loss', torchmetrics.MeanMetric),
                ('nb_loss', torchmetrics.MeanMetric),
                ('mre', torchmetrics.MeanMetric)
            ]:
                setattr(self, f'{stage}_{metric_name}', metric_cls())

    def forward(self, masked_input):
        B = masked_input.size(0)
        gene_feats = self.embedding_layer(self.gene_ids, masked_input, self.go_features, self.pos_enc)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        processed_sequence = self.encoder(gene_feats, cls_tokens, self.edge_index, self.edge_attr, batch_size=B)
        cell_repr = processed_sequence[:, 0]
        gene_out = processed_sequence[:, 1:]
        return gene_out, cell_repr

    def _compute_losses(self, gene_out, mask, counts, stage='train'):
        log_mu, log_r = self.nb_head(gene_out)
        nb = neg_binom_loss(counts, log_mu, log_r, mask=mask)
        mre = masked_relative_error(counts, torch.exp(log_mu), mask)
        getattr(self, f'{stage}_nb_loss').update(nb.detach())
        getattr(self, f'{stage}_mre').update(mre.detach())
        return nb, mre

    def _parse_batch(self, batch, stage):
        contrastive = self.hparams.config.get('use_contrastive_loss', False)
        view2 = None
        if contrastive:
            if isinstance(batch, (list, tuple)) and len(batch) == 2 and all(isinstance(b, (list, tuple)) for b in batch):
                (m1, mk1, _lab1, c1), (m2, mk2, _lab2, c2) = batch
                view2 = (m2, mk2, c2)
                return (m1, mk1, c1), view2, True
            else:
                contrastive = False  # fallback
        # single view formats
        if isinstance(batch, (list, tuple)):
            if len(batch) == 4:
                m, mk, _lab, c = batch
                return (m, mk, c), None, False
            first = batch[0]
            if isinstance(first, (list, tuple)) and len(first) == 4:
                m, mk, _lab, c = first
                return (m, mk, c), None, False
        raise RuntimeError(f"Unrecognized batch format in {stage}: type={type(batch)} length={len(batch) if hasattr(batch,'__len__') else 'NA'}")

    def training_step(self, batch, batch_idx):
        (masked_input, mask, counts), view2, contrastive = self._parse_batch(batch, 'train')
        gene_out, cell_repr = self.forward(masked_input)
        nb, mre = self._compute_losses(gene_out, mask, counts, 'train')
        total = self.alpha_nb * nb + (1.0 - self.alpha_nb) * mre
        if contrastive and hasattr(self, 'contrastive_head'):
            _, cell_repr_2 = self.forward(view2[0])
            z1 = F.normalize(self.contrastive_head(cell_repr), dim=1)
            z2 = F.normalize(self.contrastive_head(cell_repr_2), dim=1)
            sim = torch.matmul(z1, z2.T) / self.hparams.config['contrastive_temperature']
            c_loss = F.cross_entropy(sim, torch.arange(sim.size(0), device=self.device))
            total = total + self.hparams.config['contrastive_weight'] * c_loss
            self.log('train_contrastive_loss', c_loss, on_step=True, prog_bar=False)
        if dist_rank() == 0 and batch_idx == 0:
            print(f"[Debug][train] contrastive={contrastive} masked_input={tuple(masked_input.shape)}")
        self.train_loss.update(total.detach())
        self.log('train_loss_step', total, on_step=True, prog_bar=True)
        return total

    def validation_step(self, batch, batch_idx):
        (masked_input, mask, counts), _view2, contrastive = self._parse_batch(batch, 'val')
        gene_out, _ = self.forward(masked_input)
        nb, mre = self._compute_losses(gene_out, mask, counts, 'val')
        total = self.alpha_nb * nb + (1.0 - self.alpha_nb) * mre
        if dist_rank() == 0 and batch_idx == 0:
            print(f"[Debug][val] contrastive={contrastive} masked_input={tuple(masked_input.shape)}")
        self.val_loss.update(total.detach())
        return total

    def _on_epoch_end(self, stage):
        avg_loss = getattr(self, f'{stage}_loss').compute()
        nb = getattr(self, f'{stage}_nb_loss').compute()
        mre = getattr(self, f'{stage}_mre').compute()
        log_prefix = 'val' if stage == 'val' else f'{stage}_'
        self.log(f'{log_prefix}_loss', avg_loss, prog_bar=True, sync_dist=True)
        if stage == 'val' and self.trainer.is_global_zero:
            print(f"\nEpoch {self.current_epoch} Validation: Loss: {avg_loss:.4f} (nb={nb:.4f}, mre={mre:.4f})")
        for metric in ['loss', 'nb_loss', 'mre']:
            getattr(self, f'{stage}_{metric}').reset()

    def on_train_epoch_end(self): self._on_epoch_end('train')
    def on_validation_epoch_end(self): self._on_epoch_end('val')

    def configure_optimizers(self):
        cfg = self.hparams['config']
        optimizer = AdamW(self.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
        scheduler = create_scheduler_with_warmup(optimizer, warmup_steps=cfg['warmup_steps'], total_steps=self.trainer.estimated_stepping_batches)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}

    @torch.no_grad()
    def get_gene_embeddings(self):
        self.eval()
        G = self.hparams.get('num_genes', self.gene_ids.size(0))
        zero_input = torch.full((1, G), PAD_TOKEN_ID, dtype=torch.long, device=self.device)
        gene_out, _ = self.forward(zero_input)
        return gene_out.squeeze(0).cpu().numpy()

# -------------------- Script entry: data prep & training ----------------

if __name__ == '__main__':
    USE_CONTRASTIVE_LEARNING = CONFIG.get('use_contrastive_loss', False)
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        raise RuntimeError("No CUDA devices visible. Requested GPUs 1,2,3 but none are available. Check nvidia-smi and CUDA_VISIBLE_DEVICES.")
    print(f"Visible CUDA devices (after forcing GPU_DEVICE_LIST={GPU_DEVICE_LIST}): {gpu_count}")
    for i in range(gpu_count):
        try:
            print(f"  Logical GPU {i}: {torch.cuda.get_device_name(i)}")
        except Exception:
            pass

    # Strategy selection: attempt DDP if multiple GPUs (Lightning can spawn processes)
    # If user launched via torchrun, should_init_dist() will be True and env:// used.
    if gpu_count > 1:
        if should_init_dist():
            if not dist.is_initialized():
                if dist_rank() == 0:
                    print("Initializing distributed process group (env://) for provided torchrun launch...")
                dist.init_process_group(backend='nccl', timeout=timedelta(minutes=30))
            strategy_name = 'DDP-env'
        else:
            # Let Lightning spawn processes automatically
            strategy_name = 'DDP-spawn'
    else:
        strategy_name = 'SingleGPU'

    if dist_rank() == 0:
        print("="*60 + f"\n🚀 Starting Foundational Gene Transformer Training\n" +
              f"Detected {gpu_count} GPUs | Strategy: {strategy_name}\n" +
              f"Contrastive Learning Enabled: {USE_CONTRASTIVE_LEARNING}\n" + "="*60)

    clear_gpu_memory()
    print_gpu_memory("Initial")
    if dist_rank() == 0: print('Loading and aligning static graph data...')
    
    adj_full = sp.load_npz(INPUT_GRAPH_PATH)
    go_all = np.load(GO_FEATURES_PATH)
    graph_genes_full = np.load(GENE_NAMES_PATH, allow_pickle=True)

    temp_dm = SingleCellDataModule(RAW_ADATA_PATH, 1, 0, 0)
    temp_dm.prepare_data()
    used_zarr = temp_dm.compact_zarr_path if os.path.exists(temp_dm.compact_zarr_path) else temp_dm.zarr_path
    if dist_rank() == 0: print(f"Data will be loaded from: {used_zarr}")
    
    adata_used = an.read_zarr(used_zarr)
    used_var_names = set(adata_used.var_names)
    sel_idx_in_graph = [i for i, g in enumerate(graph_genes_full) if g in used_var_names]
    if not sel_idx_in_graph: raise RuntimeError('No overlap between graph genes and AnnData var_names!')

    sel_indices = np.asarray(sel_idx_in_graph, dtype=np.int64)
    go_features_reduced = torch.from_numpy(go_all[sel_indices]).float()
    
    adj_csr = adj_full.tocsr()
    adj_reduced = adj_csr[sel_indices][:, sel_indices]
    
    # <<< MODIFIED: Extract edge attributes along with the index
    edge_index_reduced, edge_attr_reduced = from_scipy_sparse_matrix(adj_reduced)
    edge_attr_reduced = edge_attr_reduced.float()
    # <<< END MODIFIED

    num_genes_reduced = go_features_reduced.size(0)
    if dist_rank() == 0:
        print(f"Graph aligned to {num_genes_reduced} genes present in the dataset.")
        print(f"Extracted {edge_index_reduced.shape[1]} edges with weights.") # <<< NEW: Confirmation print

    cache_reduced = POS_ENC_CACHE.replace('.npy', f'.{num_genes_reduced}.npy')
    pos_enc_reduced = compute_laplacian_pos_enc(edge_index_reduced, num_genes_reduced, k=32, device='cpu', cache_path=cache_reduced)
    
    data_module = SingleCellDataModule(
        RAW_ADATA_PATH, **DATA_MODULE_CONFIG, gene_names_path=GENE_NAMES_PATH,
        edge_index=edge_index_reduced, num_genes=num_genes_reduced,
        use_contrastive_loss=USE_CONTRASTIVE_LEARNING
    )
    
    hparams = {'config': CONFIG, 'data_config': DATA_MODULE_CONFIG, 'num_genes': num_genes_reduced}
    model = FoundationalGeneTransformer(
        hparams, num_genes=num_genes_reduced, go_feature_dim=go_features_reduced.size(1), 
        pos_encoding_dim=pos_enc_reduced.shape[1], edge_index=edge_index_reduced,
        edge_attr=edge_attr_reduced, go_features=go_features_reduced, pos_enc=pos_enc_reduced # <<< MODIFIED: Pass edge_attr
    )
    
    # Configure strategy according to strategy_name
    if strategy_name.startswith('DDP'):
        # enable find_unused_parameters False for speed; spawn uses 'auto' backend selection
        strategy = DDPStrategy(find_unused_parameters=False, timeout=timedelta(minutes=30))
        devices = gpu_count
    else:
        strategy = 'auto'
        devices = 1
    effective_batch_size = DATA_MODULE_CONFIG["batch_size"] * devices
    if dist_rank() == 0: print(f"Effective batch size: {effective_batch_size}")
    
    # <<< NEWLY INSERTED SUMMARY BLOCK >>>
    if dist_rank() == 0:
        import pprint
        
        # --- Calculate Model Parameters ---
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("\n" + "="*80)
        print("🚀 LAUNCH SUMMARY - Foundational Gene Transformer")
        print("="*80)
        
        print("\n--- Data & Graph Summary ---")
        print(f"Loaded graph from:         {INPUT_GRAPH_PATH}")
        print(f"Aligned number of genes:   {num_genes_reduced:,}")
        print(f"Number of edges in graph:  {edge_index_reduced.shape[1]:,}")
        print(f"Edge attributes loaded:    {'Yes' if edge_attr_reduced is not None else 'No'}")
        print(f"GO features shape:         {tuple(go_features_reduced.shape)}")
        print(f"Positional encoding shape: {tuple(pos_enc_reduced.shape)}")
        print(f"Expression vocab size:     {EXPRESSION_VOCAB_SIZE} (bins)")
        print(f"Total vocab size w/ PAD/MASK: {FULL_VOCAB_SIZE}")

        print("\n--- Model & Config Summary ---")
        print(f"Model Class:               {model.__class__.__name__}")
        print(f"Total Parameters:          {total_params:,} (~{total_params/1e6:.2f} M)")
        print(f"Trainable Parameters:      {trainable_params:,} (~{trainable_params/1e6:.2f} M)")
        print("\n[MODEL CONFIG]")
        pprint.pprint(CONFIG)
        


    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=True) # Increased patience for stability
    
    if dist_rank() == 0:
        wandb_logger = WandbLogger(project='Foundational-Gene-GT-Final', config={**CONFIG, **DATA_MODULE_CONFIG, 'num_gpus': devices, 'strategy': 'ddp', 'effective_batch_size': effective_batch_size})
        checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', dirpath=CHECKPOINT_DIR, filename='best-model-loss-{epoch:02d}-{val_loss:.4f}', save_top_k=3, save_last=True)
        callbacks_list = [checkpoint_callback, early_stopping, GPUMonitorCallback()]
    else:
        wandb_logger = False
        checkpoint_callback = None
        callbacks_list = [early_stopping, GPUMonitorCallback()]

    trainer = pl.Trainer(
        logger=wandb_logger, accelerator='gpu', callbacks=callbacks_list,
        devices=list(range(devices)) if isinstance(strategy, DDPStrategy) else devices, strategy=strategy, **TRAINER_CONFIG
    )
    
    # --- TRAINING PHASE (COMMENTED OUT AS REQUESTED) ---
    # The script will now exit after printing the summary.
    # To run training, uncomment the line below.
    
    # print('\n--- Starting Training ---')
    # print_all_gpu_status()
    # print_gpu_memory("Before training")
    
    # trainer.fit(model, data_module)
    
    # print('\n--- Training complete ---')
    
    
    # --- POST-TRAINING ANALYSIS (COMMENTED OUT AS IT REQUIRES A COMPLETED RUN) ---
    # This section saves the final embeddings from the best model.
    # It will only work if training has been completed and a checkpoint file exists.
    
    # if dist_rank() == 0:
    #     print("\n" + "="*60 + "\nFINAL TRAINING SUMMARY\n" + "="*60)
    #     final_metrics = trainer.logged_metrics
    #     for metric in ['val_loss', 'val_nb_epoch', 'val_mre_epoch']:
    #         if metric in final_metrics: print(f"  {metric}: {final_metrics[metric].item():.4f}")
        
    #     print(f"\nBest model path: {checkpoint_callback.best_model_path}")

    #     if os.path.exists(checkpoint_callback.best_model_path):
    #         print("Loading best model to save gene embeddings...")
    #         best_model = FoundationalGeneTransformer.load_from_checkpoint(
    #             checkpoint_callback.best_model_path, map_location='cpu', num_genes=num_genes_reduced,
    #             go_feature_dim=go_features_reduced.size(1), pos_encoding_dim=pos_enc_reduced.shape[1],
    #             edge_index=edge_index_reduced, edge_attr=edge_attr_reduced, go_features=go_features_reduced, pos_enc=pos_enc_reduced # <<< MODIFIED
    #         )
    #         emb = best_model.get_gene_embeddings()
    #         np.save(os.path.join(OUTPUT_DIR, 'gene_embeddings_structmask_6k.npy'), emb)
    #         print(f'Saved final gene embeddings to {os.path.join(OUTPUT_DIR, "gene_embeddings_structmask.npy")}')
    #     else:
    #         print(f"Checkpoint not found: {checkpoint_callback.best_model_path}")

    print("\n--- Script finished after printing report. Training was skipped as requested. ---")
    print_all_gpu_status()
    print_gpu_memory("After setup")
    clear_gpu_memory()