import os
# Set the visible GPU device. This must be done before importing torch.
os.environ["CUDA_VISIBLE_DEVICES"] = "2" # As in your original script

import anndata as an
import numpy as np
import torch
import torch.multiprocessing as mp
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix
import gseapy as gp
from gseapy.plot import barplot
import umap
# ADDED accuracy_score to the import list
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, roc_auc_score, accuracy_score

# Avoid 'Too many open files' with DataLoader workers
mp.set_sharing_strategy('file_system')

# Import all necessary classes from your fine-tuning scripts
# Note: Ensure these file names match your project structure.
from finetune_celltype_scgpt import (
    FoundationalGeneTransformer,
    CellTypeFinetuner,
    CellTypeDataset,
    load_and_combine_adata,
)
from finetune_bachcorr_scgpt import BatchAdversarialFinetuner

# --------------------- Config (Adjust These) ---------------------
BATCH_CORR_CKPT = '/home/elenamuia/tesi/downstream_tasks/checkpoints_batchcorr_ms_subgraph/batchcorr-ms-subgraph-epoch=49-val_recon_loss=0.216.ckpt'
FINETUNED_CKPT = '/home/elenamuia/tesi/downstream_tasks/checkpoints_celltype_ms/celltype-subgraph-epoch=32-val_f1=0.716.ckpt'

# Paths to MS data and model artifacts (matching your fine-tuning scripts)
REFERENCE_ADATA_PATH = '/home/elenamuia/tesi/downstream_tasks/ms/data/c_data.h5ad'
QUERY_ADATA_PATH = '/home/elenamuia/tesi/downstream_tasks/ms/data/filtered_ms_adata.h5ad'
MS_LABEL_COL = "Factor Value[inferred cell type - authors labels]"
MS_GENE_COL = "gene_name"
BATCH_KEY = 'source'

GENE_NAMES_PATH = '/home/elenamuia/tesi/outputs/graph_gene_names_4k.npy'
GO_FEATURES_PATH = '/home/elenamuia/tesi/outputs/gene_go_features_4k.npy'
GRAPH_PATH = '/home/elenamuia/tesi/outputs/A_context_multimodal_4k.npz'
POS_ENC_CACHE = '/home/elenamuia/tesi/outputs/laplacian_pos_enc_32.npy'
EXPR_BINS_PATH = '/home/elenamuia/tesi/outputs/expression_bins.npy'

# Keys for AnnData
LABEL_KEY = 'celltype'

# Output directory
OUTPUT_VIZ_DIR = '/home/elenamuia/tesi/downstream_tasks/visualizations_hybrid_model_ms'
os.makedirs(OUTPUT_VIZ_DIR, exist_ok=True)

# General settings
BATCH_SIZE = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ATTR_MAX_CELLS_PER_CLASS = 20
ATTR_DEVICE = 'cpu' # Use CPU for attribution to avoid potential GPU OOM with Captum
# -------------------------------------------------------------------

@torch.no_grad()
def get_embeddings_and_labels(model, dataloader, device='cuda'):
    """
    Extracts cell embeddings, true labels, and raw logits from a dataloader.
    """
    model.to(device)
    model.eval()
    all_embeddings, all_labels, all_logits = [], [], []

    for batch in tqdm(dataloader, desc="Extracting embeddings and logits from test set"):
        x, y = batch
        x = x.to(device)

        batch_size = x.size(0)
        num_total_genes = model.backbone.hparams.num_genes
        pad_token_id = model.backbone.hparams.pad_token_id
        full_input = torch.full((batch_size, num_total_genes), fill_value=pad_token_id, device=x.device, dtype=torch.long)
        full_input[:, model.model_gene_indices] = x

        _, cell_repr = model.backbone(full_input)
        logits = model.classifier(cell_repr)

        all_embeddings.append(cell_repr.cpu().numpy())
        all_labels.append(y.numpy())
        all_logits.append(logits.cpu().numpy())

    return np.concatenate(all_embeddings), np.concatenate(all_labels), np.concatenate(all_logits)

# Main visualization function
def visualize():
    print("--- Setting up Data and Models ---")

    # --- 1. Load Data and Graph components ---
    bins = np.load(EXPR_BINS_PATH) if os.path.exists(EXPR_BINS_PATH) else None
    combined_adata = load_and_combine_adata(REFERENCE_ADATA_PATH, QUERY_ADATA_PATH, MS_LABEL_COL, MS_GENE_COL)
    graph_gene_names = np.load(GENE_NAMES_PATH, allow_pickle=True)
    go_features_full = torch.from_numpy(np.load(GO_FEATURES_PATH)).float()
    adj_full = sp.load_npz(GRAPH_PATH)
    pos_enc_full = torch.from_numpy(np.load(POS_ENC_CACHE)).float()
    
    # --- 2. Define Subgraph ---
    print("--- Defining subgraph based on data-graph overlap ---")
    common_genes = sorted(list(set(graph_gene_names) & set(combined_adata.var_names)))
    if not common_genes: raise RuntimeError("No common genes found.")
    
    graph_gene_to_idx = {name: i for i, name in enumerate(graph_gene_names)}
    seed_node_indices = np.array([graph_gene_to_idx[name] for name in common_genes])
    
    adj_binary = (adj_full > 0).astype(int).tocsr()
    adj_2_hop = adj_binary @ adj_binary
    one_hop_neighbors = adj_binary[seed_node_indices].sum(axis=0).nonzero()[1]
    two_hop_neighbors = adj_2_hop[seed_node_indices].sum(axis=0).nonzero()[1]
    graph_sel_idx = np.union1d(np.union1d(seed_node_indices, one_hop_neighbors), two_hop_neighbors)
    
    num_subgraph_genes = len(graph_sel_idx)
    print(f"Subgraph defined with {num_subgraph_genes} genes.")

    go_features_sub = go_features_full[graph_sel_idx]
    pos_enc_sub = pos_enc_full[graph_sel_idx]
    adj_sub = adj_full.tocsr()[graph_sel_idx, :][:, graph_sel_idx]
    edge_index_sub, edge_attr_sub = from_scipy_sparse_matrix(adj_sub)

    adata_filtered = combined_adata[:, common_genes].copy()
    
    label_categories = adata_filtered.obs[LABEL_KEY].astype('category').cat.categories
    num_classes = len(label_categories)
    batch_categories = adata_filtered.obs[BATCH_KEY].astype('category').cat.categories
    num_batches = len(batch_categories)

    # --- 3. Stratified Data Splitting ---
    print("\n--- Performing stratified split to identify the test set ---")
    all_indices = np.arange(adata_filtered.n_obs)
    labels = adata_filtered.obs[LABEL_KEY].astype('category').cat.codes.values
    
    _, test_indices, _, _ = train_test_split(
        all_indices, labels, test_size=0.15, random_state=42, stratify=labels
    )
    adata_test = adata_filtered[test_indices, :].copy()
    print(f"Identified {len(test_indices)} cells for the test set.")
    
    # --- 4. Groundtruth UMAP Visualization on Test Set ---
    print("\n--- Generating Groundtruth UMAP on the Test Set (Before Model Integration) ---")
    adata_groundtruth = adata_test.copy()
    sc.pp.normalize_total(adata_groundtruth, target_sum=1e4)
    sc.pp.log1p(adata_groundtruth)
    sc.pp.highly_variable_genes(adata_groundtruth, n_top_genes=2000, batch_key=BATCH_KEY)
    adata_groundtruth = adata_groundtruth[:, adata_groundtruth.var.highly_variable]
    sc.tl.pca(adata_groundtruth, svd_solver='arpack')
    sc.pp.neighbors(adata_groundtruth, n_pcs=30)
    sc.tl.umap(adata_groundtruth, random_state=42)
    fig_gt_umap = sc.pl.umap(adata_groundtruth, color=[LABEL_KEY, BATCH_KEY], title=[f'Groundtruth UMAP by {LABEL_KEY}', f'Groundtruth UMAP by {BATCH_KEY}'], show=False, return_fig=True, palette='tab20')
    plt.tight_layout()
    gt_umap_path = os.path.join(OUTPUT_VIZ_DIR, 'umap_groundtruth_test_set_before_model.pdf')
    fig_gt_umap.savefig(gt_umap_path, format='pdf', bbox_inches='tight')
    plt.close(fig_gt_umap)
    del adata_groundtruth

    # --- 5. Prepare Dataloader and Assemble the Hybrid Model ---
    test_dataset = CellTypeDataset(adata_filtered, test_indices, bins)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    config = {'embedding_dim': 128, 'num_heads': 2, 'num_local_layers': 2, 'num_global_layers': 4, 'dropout': 0.1}
    vocab_size = (len(bins) + 1) if bins is not None else 512
    pad_token_id = vocab_size
    
    subgraph_orig_idx_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(graph_sel_idx)}
    model_gene_indices = torch.tensor([subgraph_orig_idx_map[i] for i in seed_node_indices], dtype=torch.long)
    
    backbone_shell = FoundationalGeneTransformer(
        config=config, num_genes=num_subgraph_genes, go_feature_dim=go_features_sub.shape[1],
        pos_encoding_dim=pos_enc_sub.shape[1], edge_index=edge_index_sub, edge_attr=edge_attr_sub.float(),
        go_features=go_features_sub, pos_enc=pos_enc_sub,
        full_vocab_size=vocab_size + 2, pad_token_id=pad_token_id
    )

    print("\n--- Assembling Hybrid Model from Checkpoints ---")
    bc_model = BatchAdversarialFinetuner.load_from_checkpoint(
        BATCH_CORR_CKPT, backbone=backbone_shell, num_batches=num_batches,
        num_genes_to_reconstruct=len(common_genes), model_gene_indices=model_gene_indices,
        lr=1e-5, lambda_adv=0.1, freeze_backbone=False, strict=False
    )
    cta_model = CellTypeFinetuner.load_from_checkpoint(
        FINETUNED_CKPT, backbone=backbone_shell, num_classes=num_classes,
        model_gene_indices=model_gene_indices, strict=False
    )
    
    hybrid_model = CellTypeFinetuner(
        backbone=bc_model.backbone, num_classes=num_classes,
        model_gene_indices=model_gene_indices
    )
    hybrid_model.classifier = cta_model.classifier
    hybrid_model.to(DEVICE).eval()
    del bc_model, cta_model; torch.cuda.empty_cache()

    # --- 6. Generate Embeddings and Predictions on the TEST SET ---
    all_embeddings, y_true, logits = get_embeddings_and_labels(hybrid_model, test_dataloader, device=DEVICE)
    y_pred = np.argmax(logits, axis=1) # Get final predictions from logits
    
    # ==================== METRICS CALCULATION ====================
    print("\n--- 7. Calculating Performance Metrics on Test Set ---")
    
    # Calculate Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy on Test Set: {accuracy:.4f}")
    
    # Calculate Macro F1 Score
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"Macro F1 Score on Test Set: {macro_f1:.4f}")

    # Calculate AUC Score
    probabilities = torch.nn.functional.softmax(torch.from_numpy(logits), dim=1).numpy()
    auc_score = roc_auc_score(y_true, probabilities, multi_class='ovr')
    print(f"AUC (One-vs-Rest) on Test Set: {auc_score:.4f}")
    # ================================================================

    # --- 8. Generate Visualizations on the TEST SET ---
    # === Model Embedding UMAP Visualization ===
    print("\n--- Generating UMAP of Cell Embeddings from Hybrid Model on Test Set ---")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    embedding_2d = reducer.fit_transform(all_embeddings)
    
    fig_umap, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.scatterplot(x=embedding_2d[:, 0], y=embedding_2d[:, 1], hue=adata_test.obs[LABEL_KEY].values, s=5, ax=axes[0], alpha=0.7).set_title('UMAP by Cell Type (Test Set)')
    axes[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), markerscale=2)
    sns.scatterplot(x=embedding_2d[:, 0], y=embedding_2d[:, 1], hue=adata_test.obs[BATCH_KEY].values, s=5, ax=axes[1], alpha=0.7).set_title('UMAP by Batch (Test Set)')
    axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), markerscale=2)
    plt.tight_layout()
    umap_path = os.path.join(OUTPUT_VIZ_DIR, 'umap_model_embeddings_test_set.pdf')
    plt.savefig(umap_path, format='pdf', bbox_inches='tight')
    plt.close(fig_umap)

    # === Confusion Matrix on Test Set ===
    print("\n--- Generating Confusion Matrix on Test Set ---")
    fig_cm, ax_cm = plt.subplots(figsize=(12, 12))
    display_labels = label_categories.to_list()
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        ax=ax_cm, xticks_rotation='vertical', normalize='true', values_format='.2f',
        display_labels=display_labels,
        cmap='Blues'
    )
    ax_cm.set_title('Normalized Confusion Matrix on Test Set (Hybrid Model)')
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_VIZ_DIR, 'confusion_matrix_test_set.pdf')
    plt.savefig(cm_path, format='pdf', bbox_inches='tight')
    plt.close(fig_cm)
    
    # ========================================================================
    # ===== GENE ATTRIBUTION and GSEA ANALYSIS (NEWLY ADDED & INTEGRATED) =====
    # ========================================================================
    print("\n--- Starting Gene Attribution and GSEA on Test Set ---")
    try:
        from captum.attr import LayerIntegratedGradients
        import pandas as pd
    except ImportError:
        print("Captum or Pandas is not installed. Skipping gene attribution.")
        return

    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.register_buffer('model_gene_indices', model.model_gene_indices)
            self.num_total_genes = model.backbone.hparams.num_genes
            self.pad_token_id = model.backbone.hparams.pad_token_id

        def forward(self, x_sub_int, target_class_idx):
            batch_size = x_sub_int.size(0)
            full_input = torch.full(
                (batch_size, self.num_total_genes), fill_value=self.pad_token_id,
                device=x_sub_int.device, dtype=torch.long
            )
            full_input[:, self.model_gene_indices] = x_sub_int
            _, cell_repr = self.model.backbone(full_input)
            logits = self.model.classifier(cell_repr)
            return logits[:, target_class_idx]

    hybrid_model.to(ATTR_DEVICE)
    model_wrapper = ModelWrapper(hybrid_model)
    model_wrapper.eval()

    lig = LayerIntegratedGradients(model_wrapper, hybrid_model.backbone.embedding_layer)

    all_gsea_results = []
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    for class_idx, cell_type in enumerate(tqdm(label_categories, desc="Attributing Genes on Test Set")):
        safe_cell_type_name = cell_type.replace(' ', '_').replace('/', '_')

        try:
            # === 1. Calculate Attributions ===
            idx_in_test = np.where(adata_test.obs[LABEL_KEY] == cell_type)[0]
            if len(idx_in_test) == 0: continue
            if len(idx_in_test) > ATTR_MAX_CELLS_PER_CLASS:
                idx_in_test = np.random.choice(idx_in_test, ATTR_MAX_CELLS_PER_CLASS, replace=False)

            raw_x = adata_test.X[idx_in_test]
            if sp.issparse(raw_x): raw_x = raw_x.toarray()

            binned_x = np.digitize(raw_x, bins=bins, right=True).astype(np.int64)
            input_tensor = torch.from_numpy(binned_x).to(ATTR_DEVICE)
            baseline = torch.zeros_like(input_tensor)

            attributions = lig.attribute(
                inputs=input_tensor, baselines=baseline, additional_forward_args=(class_idx,),
                n_steps=50, attribute_to_layer_input=False, internal_batch_size=4
            )

            attributions_for_common_genes = attributions[:, model_wrapper.model_gene_indices, :]
            attr_score = attributions_for_common_genes.sum(dim=-1).abs().mean(dim=0).detach().cpu().numpy()

            df = pd.DataFrame({'gene_name': common_genes, 'attribution': attr_score})
            df = df[df['attribution'] > 0].sort_values('attribution', ascending=False)

            if df.empty:
                print(f"Skipping GSEA for {cell_type}: No genes with positive attribution scores.")
                continue

            # === 2. Run GSEA Prerank with its own error handling ===
            pre_res = None
            try:
                gene_sets = ['GO_Biological_Process_2023', 'KEGG_2021_Human', 'Reactome_2022']
                pre_res = gp.prerank(rnk=df, gene_sets=gene_sets, threads=4, seed=42, verbose=False)

            except Exception as gsea_error:
                if "No enrich terms" in str(gsea_error):
                    print(f"GSEA for '{cell_type}' did not yield any significant results. Skipping plot generation.")
                    continue
                else:
                    raise gsea_error

            # === 3. Process, Plot, and Collect Results ===
            if pre_res is not None and not pre_res.res2d.empty:
                results_df = pre_res.res2d.copy() # Use .copy() to avoid warnings

                # <<<<<<<<<<<<<<<< NEW: Add cell type and collect results >>>>>>>>>>>>>>>>
                results_df['cell_type'] = cell_type
                all_gsea_results.append(results_df)
                # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

                # Barplot generation (remains unchanged)
                if 'Adjusted P-value' not in results_df.columns:
                    if 'FDR q-val' in results_df.columns:
                        results_df = results_df.rename(columns={'FDR q-val': 'Adjusted P-value'})
                    elif 'Adj Pval' in results_df.columns:
                        results_df = results_df.rename(columns={'Adj Pval': 'Adjusted P-value'})
                    else:
                        print(f"Warning: Could not find 'Adjusted P-value' column for {cell_type}. Skipping barplot.")
                        continue
                
                all_p_values_zero = (results_df['Adjusted P-value'] == 0).all()
                plot_title = f'Top Pathways: {cell_type}'
                plot_path = os.path.join(OUTPUT_VIZ_DIR, f'gsea_test_set_{safe_cell_type_name}.pdf')

                if all_p_values_zero:
                    print(f"Plotting '{cell_type}' with a solid color due to extremely significant p-values.")
                    barplot(results_df, title=plot_title, ofname=plot_path, color='salmon')
                else:
                    barplot(results_df, title=plot_title, ofname=plot_path)

        except Exception as e:
            print(f"An error occurred during the main process for {cell_type}: {e}")
            continue

    # =========================================================================
    # <<<<<<<<<<<<<<<< NEW: Consolidate and Save Final GSEA CSV >>>>>>>>>>>>>>>>
    # =========================================================================
    if all_gsea_results:
        print("\n--- Consolidating all GSEA results into a single CSV ---")
        
        # Combine all collected dataframes into one
        final_df = pd.concat(all_gsea_results, ignore_index=True)

        # Standardize column names to match the desired output format
        # This handles different versions of gseapy
        rename_map = {
            'Term': 'term',
            'NES': 'nes',
            'P-val': 'pval',
            'FDR q-val': 'adj_pval',
            'Adj Pval': 'adj_pval',
            'Adjusted P-value': 'adj_pval', # In case we already renamed it
            'Genes': 'genes'
        }
        # Rename columns that exist in the dataframe
        final_df.rename(columns={k: v for k, v in rename_map.items() if k in final_df.columns}, inplace=True)

        # Select and reorder columns for the final CSV
        output_columns = ['cell_type', 'term', 'nes', 'pval', 'adj_pval', 'genes']
        # Filter to only include columns that actually exist after renaming
        final_columns = [col for col in output_columns if col in final_df.columns]
        
        final_df = final_df[final_columns]

        # Save the final consolidated CSV file
        gsea_summary_path = os.path.join(OUTPUT_VIZ_DIR, 'gsea_summary_by_celltype.csv')
        final_df.to_csv(gsea_summary_path, index=False)
        
        print(f"Successfully saved consolidated GSEA results to: {gsea_summary_path}")
    else:
        print("\n--- No significant GSEA results were found across any cell types to consolidate. ---")

if __name__ == '__main__':
    visualize()


# --- 7. Calculating Performance Metrics on Test Set ---
# Accuracy on Test Set: 0.7764
# Macro F1 Score on Test Set: 0.7418
# AUC (One-vs-Rest) on Test Set: 0.9814