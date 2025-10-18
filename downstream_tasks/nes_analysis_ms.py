import pandas as pd
import matplotlib.pyplot as plt
import os
from kneed import KneeLocator

# --- Configuration ---
# The single, consolidated CSV file with GSEA results for all cell types.
INPUT_CSV_PATH = '/home/elenamuia/tesi/downstream_tasks/visualizations_hybrid_model_ms/gsea_summary_by_celltype.csv'

# Directory where the diagnostic plots will be saved.
DIAGNOSTIC_OUTPUT_DIR = '/home/elenamuia/tesi/downstream_tasks/visualizations_hybrid_model_ms/nes_knee_diagnostics/'
os.makedirs(DIAGNOSTIC_OUTPUT_DIR, exist_ok=True)

# Column names from your CSV file.
CELL_TYPE_COL = 'cell_type'
FDR_COL = 'adj_pval'
NES_COL = 'nes'

# First, we apply the FDR cutoff we decided on previously.
FDR_SIGNIFICANCE_CUTOFF = 0.05
# A conventional number of top pathways to show for comparison.
TOP_N_CONVENTIONAL_CUTOFF = 10

def run_nes_diagnostics_from_summary(input_csv: str, output_dir: str):
    """
    Loads a consolidated GSEA results file, runs the NES elbow plot diagnostic for all 
    cell types, programmatically finds the best TOP_N cutoff for each, and generates 
    a summary plot.

    Returns:
        pd.DataFrame: A dataframe with the best TOP_N cutoff for each cell type.
    """
    # 1. Load and validate the consolidated data
    try:
        gsea_df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"ERROR: Input file not found at {input_csv}")
        return pd.DataFrame()

    required_cols = [CELL_TYPE_COL, FDR_COL, NES_COL]
    if not all(col in gsea_df.columns for col in required_cols):
        print(f"ERROR: CSV must contain the columns: {', '.join(required_cols)}.")
        return pd.DataFrame()

    # Automatically get the list of cell types from the data
    cell_types = gsea_df[CELL_TYPE_COL].unique()

    best_top_n_values = {}
    all_nes_series = {}

    print("--- Running Individual NES Diagnostics for All Cell Types ---")
    for cell_type in sorted(cell_types):
        print(f"\n> Processing: {cell_type}")
        
        # Filter data for the current cell type
        cell_type_data = gsea_df[gsea_df[CELL_TYPE_COL] == cell_type].copy()
        
        # 2. First, filter for statistically significant pathways
        significant_df = cell_type_data[cell_type_data[FDR_COL] < FDR_SIGNIFICANCE_CUTOFF].copy()
        
        if significant_df.empty:
            print(f"  - No significant pathways (FDR < {FDR_SIGNIFICANCE_CUTOFF}) found. Skipping NES analysis.")
            best_top_n_values[cell_type] = 0
            continue
            
        # 3. Sort by absolute NES to find the strongest enrichments
        significant_df['abs_NES'] = significant_df[NES_COL].abs()
        sorted_nes_df = significant_df.sort_values('abs_NES', ascending=False).reset_index(drop=True)
        all_nes_series[cell_type] = sorted_nes_df['abs_NES']
        
        if len(sorted_nes_df) < 3: # kneed needs at least 3 points
            print("  - WARNING: Not enough significant pathways to find an elbow. Skipping.")
            best_top_n_values[cell_type] = len(sorted_nes_df)
            continue

        # --- Find the Elbow Programmatically using kneed ---
        # For a sorted list of scores (high to low), the curve is 'convex' and the direction is 'decreasing'
        kneedle = KneeLocator(
            x=sorted_nes_df.index, 
            y=sorted_nes_df['abs_NES'].values, 
            curve='convex', 
            direction='decreasing',
            S=1.0 # Sensitivity parameter, 1.0 is a good default
        )
        
        # --- Store the results ---
        # The elbow gives us the rank, which is our data-driven "Top N"
        # kneedle.elbow is an index, so add 1 to get the count
        best_n = kneedle.elbow + 1 if kneedle.elbow is not None else None
        best_top_n_values[cell_type] = best_n
        if best_n:
            print(f"  - Data-driven elbow found at rank {kneedle.elbow}. Recommending TOP N = {best_n}.")
        else:
            print("  - No elbow could be automatically detected. Convention is likely best.")

        # --- Generate Individual Diagnostic Plot ---
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(12, 7))
        plt.plot(sorted_nes_df.index, sorted_nes_df['abs_NES'], marker='.', linestyle='-', markersize=4, label='Sorted |NES| of significant pathways')
        plt.title(f'Absolute NES Elbow Plot for {cell_type}', fontsize=16, fontweight='bold')
        plt.xlabel('Pathway Rank (Sorted by Absolute NES)', fontsize=12)
        plt.ylabel('Absolute Normalized Enrichment Score (|NES|)', fontsize=12)
        
        # Plot conventional Top N cutoff
        plt.axvline(x=TOP_N_CONVENTIONAL_CUTOFF - 1, color='red', linestyle='--', label=f'Conventional Cutoff (Top {TOP_N_CONVENTIONAL_CUTOFF})')
            
        # Plot the DETECTED elbow
        if best_n:
            plt.axvline(x=kneedle.elbow, color='purple', linestyle=':', linewidth=2.5, label=f'Detected Elbow (Top {best_n})')
        
        plt.grid(True, which="both", ls="--", alpha=0.4)
        plt.legend()
        
        safe_cell_name = cell_type.replace(' ', '_').replace('/', '_')
        plot_path = os.path.join(output_dir, f'diagnostic_nes_elbow_{safe_cell_name}.pdf')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()

    # --- Generate the Summary Diagnostic Plot ---
    print("\n--- Generating Summary NES Diagnostic Plot ---")
    plt.figure(figsize=(14, 9))
    for cell_type, nes_series in all_nes_series.items():
        plt.plot(nes_series.index, nes_series.values, label=cell_type, alpha=0.8)

    plt.title('Summary of Absolute NES Distributions Across All Cell Types', fontsize=18, fontweight='bold')
    plt.xlabel('Pathway Rank (Sorted by Absolute NES)', fontsize=12)
    plt.ylabel('Absolute Normalized Enrichment Score (|NES|)', fontsize=12)
    plt.axvline(x=TOP_N_CONVENTIONAL_CUTOFF - 1, color='red', linestyle='--', label=f'Conventional Cutoff (Top {TOP_N_CONVENTIONAL_CUTOFF})')
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Cell Types")
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    # Limit x-axis for better visibility of the "elbow" region, e.g., show the top 50
    plt.xlim(-2, 50) 
    
    summary_plot_path = os.path.join(output_dir, 'diagnostic_nes_summary_plot.pdf')
    plt.savefig(summary_plot_path, bbox_inches='tight')
    plt.close()
    print(f"  > Summary plot saved to: {os.path.basename(summary_plot_path)}")
    
    # --- Return the summary table of best values ---
    summary_df = pd.DataFrame(best_top_n_values.items(), columns=['Cell Type', 'Data-Driven Top N'])
    return summary_df.sort_values('Data-Driven Top N', ascending=False, na_position='last').reset_index(drop=True)

# --- Main Execution Block ---
if __name__ == '__main__':
    results_df = run_nes_diagnostics_from_summary(INPUT_CSV_PATH, DIAGNOSTIC_OUTPUT_DIR)
    
    if not results_df.empty:
        print("\n\n--- Data-Driven TOP N Recommendation Results ---")
        print(results_df.to_string())
        
        print(f"\n--- Diagnostic study complete! ---")
        print(f"Check individual and summary plots in: {DIAGNOSTIC_OUTPUT_DIR}")
    else:
        print("\n--- Diagnostic study could not be completed due to errors. ---")