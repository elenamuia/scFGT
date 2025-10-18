import pandas as pd
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import scipy.io  # Add this import for mmread
from build_mnn import build_mnn_graph
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import os
import glob
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/home/elenamuia/tesi/data/PBMCs/'
PPI_PATH = '/home/elenamuia/tesi/data/9606.protein.links.v11.5.txt'
OUTPUT_DIR = '/home/elenamuia/tesi/outputs/'
PPI_DIR = '/home/elenamuia/tesi/data/'

def calculate_weighted_correlation_optimized(X_hvg: sp.csr_matrix, F_marsgt: np.ndarray) -> np.ndarray:
    """
    Calculates the cell-similarity-weighted gene correlation using sparse matrix operations
    to avoid high memory consumption.
    """
    print("\n--- Calculating Weighted Gene Correlation (Memory-Efficient Sparse Method) ---")
    
    # OPTIMIZED: Use float32 to reduce memory usage
    F_normalized = normalize(F_marsgt.astype(np.float32), norm='l2', axis=1)
    
    # OPTIMIZED: More efficient mean calculation for sparse matrices
    gene_means = np.array(X_hvg.mean(axis=0)).flatten()
    
    X_centered = X_hvg - gene_means
    
    # OPTIMIZED: Process in chunks to reduce memory usage for very large datasets
    chunk_size = min(1000, F_normalized.shape[1])
    weighted_cov = np.zeros((X_hvg.shape[1], X_hvg.shape[1]), dtype=np.float32)
    
    print(f"Processing correlation in chunks of size {chunk_size}...")
    for i in range(0, F_normalized.shape[1], chunk_size):
        end_idx = min(i + chunk_size, F_normalized.shape[1])
        F_chunk = F_normalized[:, i:end_idx]
        intermediate = X_centered.T @ F_chunk
        weighted_cov += intermediate @ intermediate.T
    
    # Compute correlation
    d = np.sqrt(np.diag(weighted_cov))
    d[d == 0] = 1
    weighted_corr = weighted_cov / np.outer(d, d)
    
    return np.nan_to_num(np.clip(weighted_corr, -1.0, 1.0))

def load_enhancer_gene_regulatory_networks():
    """
    Load and combine all enhancer-gene regulatory network (EGRN) files.
    """
    print("\n--- Loading Enhancer-Gene Regulatory Networks ---")
    
    egrn_files = glob.glob(os.path.join(DATA_DIR, 'Dataframe/*_egrn.csv'))
    print(f"Found {len(egrn_files)} EGRN files")
    
    all_egrns = []
    for file in egrn_files:
        print(f"Loading {os.path.basename(file)}...")
        df = pd.read_csv(file)
        
        # Extract sample information from filename
        filename = os.path.basename(file)
        if filename.startswith('1_'):
            sample_type = 'patient_1'
        elif filename.startswith('9_'):
            sample_type = 'patient_9'
        elif filename.startswith('12_'):
            sample_type = 'patient_12'
        else:
            sample_type = 'unknown'
        
        df['sample'] = sample_type
        df['sample_file'] = filename
        all_egrns.append(df)
    
    combined_egrn = pd.concat(all_egrns, ignore_index=True)
    print(f"Combined EGRN contains {len(combined_egrn)} peak-gene regulatory links")
    
    return combined_egrn

def filter_high_confidence_regulatory_links(egrn_df, score_threshold=1e-8):
    """
    Filter EGRN for high-confidence peak-gene regulatory relationships.
    """
    print(f"\n--- Filtering High-Confidence Regulatory Links (threshold: {score_threshold}) ---")
    
    # Convert scores to absolute values and filter
    egrn_df['abs_score'] = np.abs(egrn_df['score'])
    high_conf_egrn = egrn_df[egrn_df['abs_score'] >= score_threshold].copy()
    
    print(f"Retained {len(high_conf_egrn)} high-confidence links from {len(egrn_df)} total")
    print(f"Score range: {high_conf_egrn['abs_score'].min():.2e} to {high_conf_egrn['abs_score'].max():.2e}")
    
    return high_conf_egrn

def load_peak_accessibility_data():
    """
    Load ATAC-seq peak accessibility data.
    """
    print("\n--- Loading Peak Accessibility Data ---")
    
    # Load peak-cell matrix
    peak_cell_path = os.path.join(DATA_DIR, 'Peak_Cell.mtx')
    peak_cell_matrix = scipy.io.mmread(peak_cell_path).T.tocsr()  # Transpose to get cells x peaks
    
    # Load peak names
    peak_names_path = os.path.join(DATA_DIR, 'Peak_names.tsv')
    peak_names = pd.read_csv(peak_names_path, header=None, sep='\t')[0].values
    
    # Load cell names
    cell_names_path = os.path.join(DATA_DIR, 'Cell_names.tsv')
    cell_names = pd.read_csv(cell_names_path, header=None, sep='\t')[0].values
    
    print(f"Loaded peak accessibility: {peak_cell_matrix.shape[0]} cells x {peak_cell_matrix.shape[1]} peaks")
    
    return peak_cell_matrix, peak_names, cell_names

def create_multimodal_gene_graph(adata_hvg, ppi_interactions, egrn_data, alpha_ppi=0.7, alpha_egrn=0.3):
    """
    Create a multi-modal gene graph combining PPI and regulatory information.
    """
    print("\n--- Creating Multi-Modal Gene Graph ---")
    
    # Get gene names
    gene_names = [gene.upper() for gene in adata_hvg.var_names]
    gene_to_idx = {name: i for i, name in enumerate(gene_names)}
    n_genes = len(gene_names)
    
    # 1. Create PPI adjacency matrix (existing logic)
    print("Building PPI component...")
    ppi_rows, ppi_cols, ppi_weights = [], [], []
    for _, row in ppi_interactions.iterrows():
        gene1, gene2 = row['symbol1'], row['symbol2']
        if gene1 in gene_to_idx and gene2 in gene_to_idx:
            idx1, idx2 = gene_to_idx[gene1], gene_to_idx[gene2]
            weight = row['combined_score'] / 1000.0  # Normalize to 0-1
            ppi_rows.extend([idx1, idx2])
            ppi_cols.extend([idx2, idx1])
            ppi_weights.extend([weight, weight])
    
    A_ppi = sp.csr_matrix((ppi_weights, (ppi_rows, ppi_cols)), shape=(n_genes, n_genes))
    
    # 2. Create regulatory adjacency matrix
    print("Building regulatory component...")
    reg_rows, reg_cols, reg_weights = [], [], []
    
    # Group EGRN by gene pairs and aggregate scores
    gene_pairs = {}
    for _, row in egrn_data.iterrows():
        gene = row['gene'].upper()
        if gene in gene_to_idx:
            # For now, we'll create gene-gene regulatory links by finding genes
            # that are regulated by the same peaks (co-accessibility)
            pass
    
    # Alternative: Create gene regulatory network based on shared peak regulation
    peak_gene_map = {}
    for _, row in egrn_data.iterrows():
        peak = row['peak']
        gene = row['gene'].upper()
        score = abs(row['score'])
        
        if gene in gene_to_idx:
            if peak not in peak_gene_map:
                peak_gene_map[peak] = []
            peak_gene_map[peak].append((gene, score))
    
    # Create gene-gene regulatory links based on shared peak regulation
    for peak, gene_scores in peak_gene_map.items():
        if len(gene_scores) > 1:  # Peak regulates multiple genes
            for i, (gene1, score1) in enumerate(gene_scores):
                for j, (gene2, score2) in enumerate(gene_scores[i+1:], i+1):
                    if gene1 != gene2 and gene1 in gene_to_idx and gene2 in gene_to_idx:
                        idx1, idx2 = gene_to_idx[gene1], gene_to_idx[gene2]
                        # Weight based on geometric mean of scores
                        weight = np.sqrt(score1 * score2)
                        reg_rows.extend([idx1, idx2])
                        reg_cols.extend([idx2, idx1])
                        reg_weights.extend([weight, weight])
    
    if reg_rows:
        # Normalize regulatory weights
        max_reg_weight = max(reg_weights) if reg_weights else 1.0
        reg_weights = [w / max_reg_weight for w in reg_weights]
        A_reg = sp.csr_matrix((reg_weights, (reg_rows, reg_cols)), shape=(n_genes, n_genes))
    else:
        A_reg = sp.csr_matrix((n_genes, n_genes))
    
    print(f"PPI edges: {A_ppi.nnz // 2}")
    print(f"Regulatory edges: {A_reg.nnz // 2}")
    
    # 3. Combine with weighted sum
    A_multimodal = alpha_ppi * A_ppi + alpha_egrn * A_reg
    
    return A_multimodal, A_ppi, A_reg

def create_enhanced_cell_metadata():
    """
    Create cell metadata including sample information and predicted cell types.
    """
    print("\n--- Creating Enhanced Cell Metadata ---")
    
    # Load cell names
    cell_names_path = os.path.join(DATA_DIR, 'Cell_names.tsv')
    cell_names = pd.read_csv(cell_names_path, header=None, sep='\t')[0].values
    
    # Create basic metadata
    metadata = pd.DataFrame({
        'cell_id': cell_names,
        'sample_type': 'unknown',  # Will be filled based on analysis
        'batch': 'batch_1'         # Can be determined from processing batches
    })
    
    # Define cell type marker genes for later annotation
    cell_type_markers = {
        'T_cells': ['CD3D', 'CD3E', 'CD4', 'CD8A', 'CD8B'],
        'B_cells': ['CD19', 'MS4A1', 'CD20', 'PAX5'],
        'Monocytes': ['CD14', 'FCGR3A', 'CD16', 'LYZ'],
        'NK_cells': ['KLRD1', 'KLRB1', 'NCR1', 'NKG7'],
        'Dendritic_cells': ['FCER1A', 'CST3', 'CLEC9A'],
        'Platelets': ['PPBP', 'PF4', 'TUBB1']
    }
    
    print(f"Created metadata for {len(metadata)} cells")
    return metadata, cell_type_markers

# Main execution
print("=== ENHANCED MULTI-MODAL DATA UPLOAD ===")

# Load existing RNA data
h5ad_file = os.path.join(DATA_DIR, 'ycpu.h5ad')
adata_rna_raw = sc.read_h5ad(h5ad_file, backed='r')
print(f"RNA data loaded: {adata_rna_raw.shape}")

# Load MarsGT embeddings
initial_features_marsgt_df = pd.read_csv(
    os.path.join(DATA_DIR, "cell_emb10.csv"), 
    header=0, 
    index_col=0
)
print(f"MarsGT embeddings loaded: {initial_features_marsgt_df.shape}")

# Data Alignment Phase
common_cells = initial_features_marsgt_df.index.intersection(adata_rna_raw.obs_names)
print(f"Found {len(common_cells)} common cells")

if len(common_cells) < 100:
    print(f"WARNING: Only {len(common_cells)} common cells found.")

adata_rna = adata_rna_raw[common_cells, :]
adata_rna = adata_rna.to_memory()
initial_features_marsgt = initial_features_marsgt_df.loc[common_cells].to_numpy()

# Preprocessing
print("\nPreprocessing RNA data...")
sc.pp.filter_genes(adata_rna, min_cells=481)
sc.pp.highly_variable_genes(adata_rna, n_top_genes=6000, flavor='seurat_v3')

adata_hvg = adata_rna[:, adata_rna.var['highly_variable']].copy()
print(f"Using {adata_hvg.shape[1]} highly variable genes")
sc.pp.normalize_total(adata_hvg, target_sum=1e4)
sc.pp.log1p(adata_hvg)

# Build cell scaffold (existing)
A_cell_scaffold = build_mnn_graph(initial_features_marsgt, k=20)
sp.save_npz(os.path.join(OUTPUT_DIR, 'A_cell_scaffold_6k.npz'), A_cell_scaffold)

# === NEW: Multi-modal enhancements ===

# 1. Load EGRN data
egrn_data = load_enhancer_gene_regulatory_networks()
high_conf_egrn = filter_high_confidence_regulatory_links(egrn_data, score_threshold=1e-10)

# 2. Load peak accessibility
peak_matrix, peak_names, peak_cell_names = load_peak_accessibility_data()

# 3. Load and process PPI data
print("\n--- Loading PPI Data ---")
ppi_df = pd.read_csv(os.path.join(PPI_DIR, '9606.protein.links.v11.5.txt'), sep=' ', header=0)
info_df = pd.read_csv(os.path.join(PPI_DIR, '9606.protein.info.v11.5.txt'), sep='\t')
ppi_df.columns = ['protein1', 'protein2', 'combined_score']

# Map PPI to gene symbols
ensp_to_symbol_map = pd.Series(
    info_df.preferred_name.str.upper().values,
    index=info_df['#string_protein_id']
).dropna()

ppi_df['symbol1'] = ppi_df['protein1'].map(ensp_to_symbol_map)
ppi_df['symbol2'] = ppi_df['protein2'].map(ensp_to_symbol_map)
ppi_df.dropna(subset=['symbol1', 'symbol2'], inplace=True)

# Filter for HVGs
hvg_symbols = set(gene.upper() for gene in adata_hvg.var_names)
ppi_hvg = ppi_df[
    ppi_df['symbol1'].isin(hvg_symbols) &
    ppi_df['symbol2'].isin(hvg_symbols) &
    (ppi_df['combined_score'] >= 400)
]

# 4. Create multi-modal gene graph
A_multimodal, A_ppi, A_reg = create_multimodal_gene_graph(
    adata_hvg, ppi_hvg, high_conf_egrn, 
    alpha_ppi=0.7, alpha_egrn=0.3
)

# 5. Apply correlation weighting
if not isinstance(adata_hvg.X, sp.csr_matrix):
    adata_hvg.X = adata_hvg.X.tocsr()

weighted_corr = calculate_weighted_correlation_optimized(
    X_hvg=adata_hvg.X,
    F_marsgt=initial_features_marsgt
)

# 6. Create final contextualized graphs
A_context_multimodal = A_multimodal.multiply(np.abs(weighted_corr))
A_context_ppi_only = A_ppi.multiply(np.abs(weighted_corr))
A_context_reg_only = A_reg.multiply(np.abs(weighted_corr))

# 7. Create cell metadata
cell_metadata, cell_type_markers = create_enhanced_cell_metadata()

# Save all outputs
print("\n--- Saving Enhanced Outputs ---")

# Save gene graphs
sp.save_npz(os.path.join(OUTPUT_DIR, 'A_context_multimodal_6k.npz'), A_context_multimodal)
sp.save_npz(os.path.join(OUTPUT_DIR, 'A_context_ppi_only_6k.npz'), A_context_ppi_only)
sp.save_npz(os.path.join(OUTPUT_DIR, 'A_context_regulatory_only_6k.npz'), A_context_reg_only)

# Save gene names
graph_gene_names = np.array([gene.upper() for gene in adata_hvg.var_names])
np.save(os.path.join(OUTPUT_DIR, 'graph_gene_names_6k.npy'), graph_gene_names)

# Save EGRN data
high_conf_egrn.to_csv(os.path.join(OUTPUT_DIR, 'high_confidence_egrn_6k.csv'), index=False)

# Save cell metadata
cell_metadata.to_csv(os.path.join(OUTPUT_DIR, 'cell_metadata_6k.csv'), index=False)

# Save cell type markers
pd.DataFrame([(marker, cell_type) for cell_type, markers in cell_type_markers.items() 
              for marker in markers], 
             columns=['gene', 'cell_type']).to_csv(
    os.path.join(OUTPUT_DIR, 'cell_type_markers_6k.csv'), index=False
)

# Save peak data
np.save(os.path.join(OUTPUT_DIR, 'peak_names_6k.npy'), peak_names)
sp.save_npz(os.path.join(OUTPUT_DIR, 'peak_cell_matrix_6k.npz'), peak_matrix)

print(f"\n=== ENHANCED MULTI-MODAL DATA PROCESSING COMPLETE ===")
print(f"Multi-modal gene graph: {A_context_multimodal.shape}")
print(f"PPI-only graph: {A_context_ppi_only.shape}")  
print(f"Regulatory-only graph: {A_context_reg_only.shape}")
print(f"Peak accessibility: {peak_matrix.shape}")
print(f"Regulatory links: {len(high_conf_egrn)}")
print(f"Cell metadata: {len(cell_metadata)}")
