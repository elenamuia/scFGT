import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import re
import textwrap
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec

# --- Configuration ---
# The single, consolidated CSV file with GSEA results for all cell types.
INPUT_CSV_PATH = '/home/elenamuia/tesi/downstream_tasks/visualizations_hybrid_model_ms/gsea_summary_by_celltype.csv'

# Directory where the final dot plots will be saved.
OUTPUT_DIR = '/home/elenamuia/tesi/downstream_tasks/visualizations_hybrid_model_ms/gsea_dot_plots/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Column names from your CSV file.
CELL_TYPE_COL = 'cell_type'
TERM_COL = 'term'
FDR_COL = 'adj_pval'
NES_COL = 'nes'

# Analysis Parameters
FDR_CUTOFF = 0.05
TOP_N_PATHWAYS_INDIVIDUAL = 15  # Max pathways to show on individual cell type plots
TOP_N_PER_CELL_TYPE_SUMMARY = 5 # Pathways per cell type to consider for the summary plot

# --- Helper Function to Clean Pathway Names ---
def clean_term(term, max_width=50):
    """
    Cleans and formats GSEA term names for plotting.
    """
    # Remove GO/KEGG codes, e.g., (GO:0042110)
    term = re.sub(r'\s\([A-Z0-9:]+\)$', '', term, flags=re.IGNORECASE)
    # Remove common database prefixes
    prefixes_to_remove = ['REACTOME_', 'GOBP_', 'GOCC_', 'KEGG_', 'HALLMARK_', 'BIOCARTA_', 'WP_']
    for prefix in prefixes_to_remove:
        if term.upper().startswith(prefix):
            term = term[len(prefix):]
    # Remove repetitive phrases
    phrases_to_remove = ['positive regulation of ', 'regulation of ']
    for phrase in phrases_to_remove:
        if re.match(phrase, term, re.IGNORECASE):
            term = term[len(phrase):]
    # Standardize format
    term = term.replace('_', ' ').capitalize()
    return '\n'.join(textwrap.wrap(term, width=max_width, break_long_words=False))


# --- FUNCTION 1: Plotting Individual Cell Type Results ---
def plot_individual_cell_type(cell_type: str, gsea_df: pd.DataFrame, output_dir: str):
    """
    Filters data for a single cell type and creates a dot-heatmap plot.
    """
    print(f"--- Processing Individual Plot for: {cell_type} ---")
    
    # Filter for the specific cell type and for significance
    significant_df = gsea_df[(gsea_df[CELL_TYPE_COL] == cell_type) & (gsea_df[FDR_COL] < FDR_CUTOFF)].copy()

    if significant_df.empty:
        print(f"  > No significant pathways (FDR < {FDR_CUTOFF}) found. Skipping plot.")
        return

    # Get top N upregulated and top N downregulated pathways
    top_up = significant_df[significant_df[NES_COL] > 0].nlargest(TOP_N_PATHWAYS_INDIVIDUAL, NES_COL)
    top_down = significant_df[significant_df[NES_COL] < 0].nsmallest(TOP_N_PATHWAYS_INDIVIDUAL, NES_COL)
    plot_df = pd.concat([top_up, top_down])

    if plot_df.empty:
        print(f"  > No pathways to plot for {cell_type}.")
        return
    
    plot_df['Term_cleaned'] = plot_df[TERM_COL].apply(lambda x: clean_term(x, max_width=60))
    plot_df['-log10(FDR)'] = -np.log10(plot_df[FDR_COL].replace(0, 1e-300)) # Handle exact zero p-values
    plot_df = plot_df.sort_values(NES_COL)

    # --- Plotting Setup ---
    plot_height = max(7, len(plot_df) * 0.4)
    fig = plt.figure(figsize=(12, plot_height))
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.3)
    ax = fig.add_subplot(gs[0])
    legend_ax = fig.add_subplot(gs[1])

    scatter = ax.scatter(data=plot_df, x=NES_COL, y='Term_cleaned', c=NES_COL, 
                         s=plot_df['-log10(FDR)'] * 20, # Adjust dot size scaling
                         cmap='vlag', edgecolor='black', linewidth=0.5)
    
    # --- Aesthetics ---
    ax.set_title(f'Enriched Pathways in {cell_type.replace("_", " ")}', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Normalized Enrichment Score (NES)', fontsize=12)
    ax.set_ylabel('')
    ax.grid(axis='y', linestyle='-', alpha=0.6)
    ax.axvline(0, color='grey', linestyle='--', zorder=0)

    # --- Legends ---
    cbar = fig.colorbar(scatter, ax=ax, orientation='vertical', pad=0.02, aspect=40)
    cbar.set_label('Normalized Enrichment Score (NES)', size=12, weight='bold')

    size_values = np.percentile(plot_df['-log10(FDR)'], [10, 30, 60, 90]).round(1)
    # Ensure size values are unique to avoid legend errors
    size_values = sorted(list(set(size_values))) 
    size_handles = [mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                                 markersize=np.sqrt(s*20), label=s) for s in size_values]
    legend_ax.legend(handles=size_handles, title='-log10(FDR)', loc='center left', 
                     frameon=True, edgecolor='black', labelspacing=2.0)
    legend_ax.get_legend().get_title().set_fontweight('bold')
    legend_ax.axis('off')

    # --- Save Plot ---
    safe_cell_name = cell_type.replace(' ', '_').replace('/', '_')
    plot_path = os.path.join(output_dir, f'dot_heatmap_gsea_{safe_cell_name}.pdf')
    plt.savefig(plot_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  > Individual plot saved to: {os.path.basename(plot_path)}")


# --- FUNCTION 2: Creating the Summary Plot (ADAPTED FOR SPARSE DATA) ---
def create_summary_plot(gsea_df: pd.DataFrame, output_dir: str):
    """
    Aggregates GSEA results and creates a summary dot-heatmap plot that is
    dynamically sized and scaled to fit the available data.
    """
    print("\n\n--- Creating Summary Plot ---")
    
    # Filter for significant results globally
    significant_df = gsea_df[gsea_df[FDR_COL] < FDR_CUTOFF].copy()
    if significant_df.empty:
        print("  > No significant results found across all cell types. Cannot create summary plot.")
        return

    print(f"  > Selecting top {TOP_N_PER_CELL_TYPE_SUMMARY} pathways per cell type (by lowest FDR)...")
    top_pathways = significant_df.groupby(CELL_TYPE_COL, group_keys=False).apply(
        lambda x: x.nsmallest(TOP_N_PER_CELL_TYPE_SUMMARY, FDR_COL)
    ).reset_index(drop=True)
    
    unique_top_terms = top_pathways[TERM_COL].unique()
    if len(unique_top_terms) == 0:
        print("  > No pathways met the criteria for the summary plot.")
        return
        
    print(f"  > Found {len(unique_top_terms)} unique top pathways to display.")
    
    # Filter the main significant df to only include these top pathways
    plot_df = significant_df[significant_df[TERM_COL].isin(unique_top_terms)].copy()
    
    # <<< FIX 1: DATA PREPARATION FOR PLOTTING >>>
    plot_df['Term_cleaned'] = plot_df[TERM_COL].apply(lambda x: clean_term(x, max_width=45))
    plot_df['-log10(FDR)'] = -np.log10(plot_df[FDR_COL].replace(0, 1e-300))
    
    # <<< FIX 2: LOGARITHMIC SCALING FOR DOT SIZES >>>
    # This prevents giant dots from overpowering the plot.
    # The '10 *' is a scaling factor that can be tuned.
    plot_df['dot_size'] = np.log1p(plot_df['-log10(FDR)']) * 30
    
    # Sort data for organized plotting. This is crucial for categorical plots.
    # We get the order from the data itself.
    cell_type_order = sorted(plot_df[CELL_TYPE_COL].unique())
    term_order = sorted(plot_df['Term_cleaned'].unique(), reverse=True)
    
    plot_df[CELL_TYPE_COL] = pd.Categorical(plot_df[CELL_TYPE_COL], categories=cell_type_order, ordered=True)
    plot_df['Term_cleaned'] = pd.Categorical(plot_df['Term_cleaned'], categories=term_order, ordered=True)
    
    print("  > Generating the summary plot figure...")
    
    # <<< FIX 3: DYNAMIC PLOT SIZING >>>
    # Calculate dimensions based on the amount of data
    num_pathways = len(term_order)
    num_cell_types = len(cell_type_order)
    plot_height = max(6, num_pathways * 0.6)  # Base height + per-pathway height
    plot_width = max(8, num_cell_types * 1.5) # Base width + per-celltype width

    fig = plt.figure(figsize=(plot_width, plot_height))
    gs = gridspec.GridSpec(1, 2, width_ratios=[8, 1], wspace=0.1) 
    ax = fig.add_subplot(gs[0])
    legend_ax = fig.add_subplot(gs[1])
    
    nes_max_abs = abs(plot_df[NES_COL]).max() * 1.05
    
    # --- Main Scatter Plot ---
    scatter = ax.scatter(
        x=plot_df[CELL_TYPE_COL], y=plot_df['Term_cleaned'], c=plot_df[NES_COL],
        s=plot_df['dot_size'], # Use the new scaled dot size
        cmap='vlag', edgecolor='black',
        linewidth=0.5, vmin=-nes_max_abs, vmax=nes_max_abs
    )

    # --- Aesthetics and Formatting ---
    ax.set_title('Summary of GSEA Enriched Pathways Across MS Cell Types', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    ax.tick_params(axis='x', labelrotation=45, labelsize=11)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey', zorder=0)
    
    # --- Legends ---
    cbar = fig.colorbar(scatter, ax=ax, pad=0.01, aspect=30, shrink=0.9)
    cbar.set_label('Normalized Enrichment Score (NES)', size=11, weight='bold')
    
    # <<< FIX 4: UPDATED SIZE LEGEND FOR LOG SCALING >>>
    # Create legend handles based on the original -log10(FDR) values
    original_size_values = np.percentile(plot_df['-log10(FDR)'], [10, 50, 90]).round(1)
    original_size_values = sorted(list(set(p for p in original_size_values if p > 0)))
    
    size_handles = [mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                                 # Calculate the marker size using the same log transform
                                 markersize=np.sqrt(np.log1p(s) * 30), 
                                 label=f'{s:.1f}') for s in original_size_values]
                                 
    if size_handles:
        legend_ax.legend(handles=size_handles, title='-log10(FDR)',
                         loc='center left', frameon=True,
                         edgecolor='black', labelspacing=2.0)
        legend_ax.get_legend().get_title().set_fontweight('bold')
    
    legend_ax.axis('off')

    fig.tight_layout()

    # --- Save Plot ---
    summary_plot_path = os.path.join(output_dir, 'summary_gsea_dot_heatmap_ms.pdf')
    plt.savefig(summary_plot_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    
    print(f"--- Summary plot saved successfully to: {summary_plot_path} ---")
# --- Main Execution Block ---
if __name__ == '__main__':
    # Load the consolidated data once
    try:
        master_gsea_df = pd.read_csv(INPUT_CSV_PATH)
        print(f"Successfully loaded {len(master_gsea_df)} GSEA results from consolidated file.")
    except FileNotFoundError:
        print(f"FATAL ERROR: Input file not found at {INPUT_CSV_PATH}")
        exit()

    # Get cell types automatically from the data
    cell_types_in_data = master_gsea_df[CELL_TYPE_COL].unique()

    # First, run the loop to generate a plot for each cell type
    for cell_type_name in sorted(cell_types_in_data):
        plot_individual_cell_type(cell_type_name, master_gsea_df, OUTPUT_DIR)
    
    # After the loop, call the function to create the summary plot
    create_summary_plot(master_gsea_df, OUTPUT_DIR)
    
    print("\n--- All processing complete! ---")
    print(f"Individual and summary plots are saved in: {OUTPUT_DIR}")