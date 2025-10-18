import pandas as pd
import matplotlib.pyplot as plt
import os
from kneed import KneeLocator # Import the KneeLocator tool

# --- Configuration ---
# The single, consolidated CSV file with GSEA results for all cell types.
INPUT_CSV_PATH = '/home/elenamuia/tesi/downstream_tasks/visualizations_hybrid_model_ms/gsea_summary_by_celltype.csv'

# Directory where the diagnostic plots will be saved.
DIAGNOSTIC_OUTPUT_DIR = '/home/elenamuia/tesi/downstream_tasks/visualizations_hybrid_model_ms/fdr_knee_diagnostics/'
os.makedirs(DIAGNOSTIC_OUTPUT_DIR, exist_ok=True)

# Column names from your CSV file.
CELL_TYPE_COL = 'cell_type'
FDR_COL = 'adj_pval'

# A standard cutoff to show for comparison.
FDR_CONVENTIONAL_CUTOFF = 0.05

def run_fdr_diagnostics_from_summary(input_csv: str, output_dir: str):
    """
    Loads a consolidated GSEA results file, runs the FDR elbow plot diagnostic for 
    all cell types, programmatically finds the best cutoff for each, and generates 
    a summary plot.

    Returns:
        pd.DataFrame: A dataframe with the best FDR cutoff for each cell type.
    """
    # 1. Load and validate the consolidated data
    try:
        gsea_df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"ERROR: Input file not found at {input_csv}")
        return pd.DataFrame()

    if CELL_TYPE_COL not in gsea_df.columns or FDR_COL not in gsea_df.columns:
        print(f"ERROR: CSV must contain '{CELL_TYPE_COL}' and '{FDR_COL}' columns.")
        return pd.DataFrame()

    # Automatically get the list of cell types from the data
    cell_types = gsea_df[CELL_TYPE_COL].unique()

    best_fdr_values = {}
    all_fdr_series = {}

    print("--- Running Individual FDR Diagnostics for All Cell Types ---")
    for cell_type in sorted(cell_types):
        print(f"\n> Processing: {cell_type}")
        
        # Filter data for the current cell type and sort by FDR
        cell_type_data = gsea_df[gsea_df[CELL_TYPE_COL] == cell_type]
        sorted_fdr = cell_type_data[FDR_COL].sort_values().reset_index(drop=True)
        all_fdr_series[cell_type] = sorted_fdr
        
        if len(sorted_fdr) < 3: # kneed needs at least 3 points
            print("  - WARNING: Not enough data points to find an elbow. Skipping.")
            best_fdr_values[cell_type] = None
            continue

        # --- Find the Elbow Programmatically using kneed ---
        # S=1.0 is a sensitivity parameter; 1.0 is a good default.
        kneedle = KneeLocator(
            x=sorted_fdr.index, 
            y=sorted_fdr.values, 
            curve='convex', 
            direction='increasing', 
            S=1.0
        )
        
        # --- Store the results ---
        if kneedle.elbow is not None:
            elbow_rank = kneedle.elbow
            best_fdr = sorted_fdr[elbow_rank]
            best_fdr_values[cell_type] = best_fdr
            print(f"  - Data-driven elbow found at rank {elbow_rank}, FDR = {best_fdr:.4f}")
        else:
            best_fdr_values[cell_type] = None
            print("  - No elbow could be automatically detected.")

        # --- Generate Individual Diagnostic Plot ---
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(12, 7))
        plt.plot(sorted_fdr.index, sorted_fdr, marker='.', linestyle='-', markersize=4, label='Sorted Adj. P-values')
        plt.title(f'FDR Elbow Plot for {cell_type}', fontsize=16, fontweight='bold')
        plt.xlabel('Pathway Rank (Sorted by FDR)', fontsize=12)
        plt.ylabel('Adjusted P-value (Log Scale)', fontsize=12)
        plt.yscale('log')
        
        # Plot conventional cutoff
        plt.axhline(y=FDR_CONVENTIONAL_CUTOFF, color='red', linestyle='--', label=f'Conventional Cutoff ({FDR_CONVENTIONAL_CUTOFF})')
        num_below_cutoff = (sorted_fdr < FDR_CONVENTIONAL_CUTOFF).sum()
        if num_below_cutoff > 0:
            plt.axvline(x=num_below_cutoff, color='green', linestyle='--', label=f'{num_below_cutoff} pathways < {FDR_CONVENTIONAL_CUTOFF}')
            
        # Plot the DETECTED elbow
        if kneedle.elbow is not None:
            plt.axvline(x=elbow_rank, color='purple', linestyle=':', linewidth=2.5, label=f'Detected Elbow (FDR ≈ {best_fdr:.3f})')
        
        plt.grid(True, which="both", ls="--", alpha=0.4)
        plt.legend()
        
        safe_cell_name = cell_type.replace(' ', '_').replace('/', '_')
        plot_path = os.path.join(output_dir, f'diagnostic_fdr_elbow_{safe_cell_name}.pdf')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()

    # --- Generate the Summary Diagnostic Plot ---
    print("\n--- Generating Summary Diagnostic Plot ---")
    plt.figure(figsize=(14, 9))
    for cell_type, fdr_series in all_fdr_series.items():
        plt.plot(fdr_series.index, fdr_series.values, label=cell_type, alpha=0.8)

    plt.title('Summary of FDR Distributions Across All Cell Types', fontsize=18, fontweight='bold')
    plt.xlabel('Pathway Rank (Sorted by FDR)', fontsize=12)
    plt.ylabel('Adjusted P-value (Log Scale)', fontsize=12)
    plt.yscale('log')
    plt.axhline(y=FDR_CONVENTIONAL_CUTOFF, color='red', linestyle='--', label=f'Conventional Cutoff ({FDR_CONVENTIONAL_CUTOFF})')
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Cell Types")
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    
    summary_plot_path = os.path.join(output_dir, 'diagnostic_fdr_summary_plot.pdf')
    plt.savefig(summary_plot_path, bbox_inches='tight')
    plt.close()
    print(f"  > Summary plot saved to: {os.path.basename(summary_plot_path)}")
    
    # --- Return the summary table of best values ---
    summary_df = pd.DataFrame(best_fdr_values.items(), columns=['Cell Type', 'Data-Driven FDR Cutoff'])
    return summary_df.sort_values('Data-Driven FDR Cutoff').reset_index(drop=True)

# --- Main Execution Block ---
if __name__ == '__main__':
    results_df = run_fdr_diagnostics_from_summary(INPUT_CSV_PATH, DIAGNOSTIC_OUTPUT_DIR)
    
    if not results_df.empty:
        print("\n\n--- Data-Driven FDR Cutoff Results ---")
        print(results_df.to_string())
        
        print(f"\n--- Diagnostic study complete! ---")
        print(f"Check individual and summary plots in: {DIAGNOSTIC_OUTPUT_DIR}")
    else:
        print("\n--- Diagnostic study could not be completed due to errors. ---")