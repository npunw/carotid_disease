# %% [markdown]
# ## Pipeline

# %%
import pandas as pd
import os
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import gseapy
import sys
import argparse
import yaml
import os
import json
from matplotlib import pyplot as plt

# %% [markdown]
# ### Loading

# %%



def prepare_categorical_variable_for_pwc(clinical_df, var, fill_value=-1, verbose=True):
    """
    Prepares a clinical variable for pairwise comparison analysis by creating binary series for each pair of unique values.

    Args:
        clinical_df (pd.DataFrame): clinical data with sample index.
        var (str): variable name.
        fill_value (int): value used for invalid/missing entries (default: -1).
        verbose (bool): print debug info.

    Yields:
        tuple: (comparison_name, pd.Series) where Series contains 0/1/-1 values for binary comparison.
    """
    series = clinical_df[var].copy().fillna(fill_value)
    
    # Get unique values excluding fill_value
    unique_vals = [val for val in series.unique() if val != fill_value]
    
    if verbose:
        print(f"{var} unique values (excluding fill_value): {unique_vals}")
    
    # Generate all pairs of unique values
    for i, val1 in enumerate(unique_vals):
        for val2 in unique_vals[i+1:]:
            comparison_name = f"({var}){val1}_vs_{val2}"
            
            # Create binary series: val1=1, val2=0, others=fill_value
            binary_series = series.copy()
            binary_series = binary_series.map({val1: 1, val2: 0}).fillna(fill_value)
            
            if verbose:
                counts = binary_series.value_counts().sort_index()
                print(f"{comparison_name}: {dict(counts)}")
            
            yield comparison_name, binary_series.astype(int)

# %% [markdown]
# ### Cleaning

# %%
def align_and_filter(protein_df, binary_series):
    """
    Joins protein data and binary label, drops samples with label == -1.
    """
    df = protein_df.join(binary_series.rename("label"))
    no_label_bool = df["label"] == -1
    print(f"Filtering out {no_label_bool.sum()}/{len(df)} samples with irrelevant labels.")
    df = df[~no_label_bool]
    
    return df.drop(columns="label"), df["label"]

# %%
def filter_features_by_missingness(X, threshold=0.2, verbose=True):
    """
    Remove protein features with too many missing values.

    Args:
        X (pd.DataFrame): samples x proteins.
        threshold (float): maximum fraction of missing values allowed (e.g., 0.2 = 20%).
        verbose (bool): print number of features removed.

    Returns:
        pd.DataFrame: filtered X with fewer columns.
    """
    missing_fraction = X.isna().mean()
    keep_cols = missing_fraction[missing_fraction <= threshold].index
    if verbose:
        dropped = len(X.columns) - len(keep_cols)
        print(f"Filtered out {dropped}/{len(X.columns)} proteins with >{threshold*100:.0f}% missing values.")
    return X[keep_cols]

# %%
def impute_missing_values(X, method="mean"):
    """
    Impute missing values in X.

    Args:
        X (pd.DataFrame): samples x proteins.
        method (str): "mean", "median", or "zero".

    Returns:
        pd.DataFrame: imputed X.
    """
    if method == "mean":
        return X.fillna(X.mean())
    elif method == "median":
        return X.fillna(X.median())
    elif method == "zero":
        return X.fillna(0)
    else:
        raise ValueError("Invalid imputation method. Choose from 'mean', 'median', or 'zero'.")

# %%
def normalize(X):
    """
    Normalize protein expression data.

    Args:
        X (pd.DataFrame): samples x proteins.

    Returns:
        pd.DataFrame: normalized X using z-score normalization.
    """
    return (X - X.mean()) / X.std()

# %% [markdown]
# ### PWC for DEPs

# %%


def pairwise_ttest(feature_df, labels):
    """
    Run Welch’s t-test for each feature (column) between two groups defined by labels.
    """
    group0 = feature_df[labels == 0]
    group1 = feature_df[labels == 1]

    stats = []
    for protein in feature_df.columns:
        stat, p = ttest_ind(group0[protein], group1[protein], equal_var=False, nan_policy="omit")
        stats.append((protein, stat, p))

    result_df = pd.DataFrame(stats, columns=["Protein", "T-stat", "P-value"]).set_index("Protein")
    result_df["FDR"] = multipletests(result_df["P-value"], method="fdr_bh")[1]
    return result_df

# %%
def get_deps(result_df, fdr_thresh=0.05):
    return result_df[result_df["FDR"] < fdr_thresh].index.tolist()

# %%
def dep_pipeline(protein_df, binary_var, missingness_thresh=0.2, imputation_method="mean", fdr_thresh=0.05):

    X, y = align_and_filter(protein_df, binary_var)
    X = filter_features_by_missingness(X, threshold=missingness_thresh)
    X = impute_missing_values(X, method=imputation_method)
    X = normalize(X)
    results = pairwise_ttest(X, y)
    deps = get_deps(results, fdr_thresh=fdr_thresh)
    print(f"Found {len(deps)} differentially expressed proteins.")
    
    return deps

# %% [markdown]
# ### Pathway Enrichment

# %%
def get_deg_list(dep_list, protgroup_gene_map):
    deg_list = [protgroup_gene_map[dep] for dep in dep_list]
    # Take the first gene in delimited string
    deg_list = [gene.split(";")[0] for gene in deg_list]
    print(f"Using {len(deg_list)} mapped genes.")
    return deg_list

# %%

def run_enrichr(gene_list, db_list, fdr=0.05):
    if not gene_list or len(gene_list) < 2:
        print("Insufficient genes for enrichment.")
        return pd.DataFrame()
    
    print(f"Running enrichment analysis against '{db_list}'")
    enr = gseapy.enrichr(gene_list=gene_list, gene_sets=db_list, organism="human", cutoff=fdr)
    return enr


# %%
def annotate_enrichr_results(enr):
    '''Add Rich Factor and Gene Ratio for plotting'''
    results_df = enr.results
    n_genes = len(enr.gene_list)
    results_df["overlap_count"] = results_df["Overlap"].str.split("/").str[0].astype(int)
    results_df["pathway_size"] = results_df["Overlap"].str.split("/").str[1].astype(int)
    results_df["rich_factor"] = results_df["overlap_count"] / results_df["pathway_size"]
    results_df["gene_ratio"] = results_df["overlap_count"] / n_genes
    results_df['neg_log10_pval'] = -np.log10(results_df['Adjusted P-value'])
    return results_df

# %%
def create_bubble_plot(enr, db_name, x_axis="rich_factor", fdr_thresh=0.05, title=None,output_path=None, figsize=(10, 6), max_terms=20):
    # Annotate results and filter for db_name
    n_genes = len(enr.gene_list)
    df = annotate_enrichr_results(enr)
    df = df[df["Gene_set"] == db_name]
    df = df[df["Adjusted P-value"] < fdr_thresh]
    df = df.sort_values('Adjusted P-value').head(max_terms)# Sort by adjusted p-value and take top terms

    # Check if df is empty
    if df.empty:
        print(f"No enriched pathways found for {db_name} with FDR < {fdr_thresh}.")
        return
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    if x_axis not in ["rich_factor", "gene_ratio"]:
        raise ValueError(f"Invalid x_axis: {x_axis}. Must be one of: ['rich_factor', 'gene_ratio']")
    
    # Create bubble plot
    scatter = ax.scatter(
        df[x_axis], 
        range(len(df)),
        s=df['Odds Ratio'] * 1,  # Scale bubble size
        c=df['neg_log10_pval'],  # Color by significance
        cmap='Reds',
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )
    
    # Customize the plot
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['Term'], fontsize=10)
    if x_axis == "rich_factor":
        ax.set_xlabel('Rich Factor \n(Overlap / Pathway)', fontsize=12)
    elif x_axis == "gene_ratio":
        ax.set_xlabel('Gene Ratio \n(Overlap / Input)', fontsize=12)
    ax.set_ylabel(f'Pathway Terms \n({db_name})', fontsize=12)

    if title is None:
        title = f'Pathway Enrichment ({n_genes} genes)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('-log10(Adjusted P-value)', fontsize=10)
    
    # Add legend for bubble sizes
    # Create sample bubbles for legend
    legend_sizes = [50, 100, 200]  # Example odds ratios
    legend_bubbles = []
    for size in legend_sizes:
        legend_bubbles.append(plt.scatter([], [], s=size*2, c='gray', alpha=0.7, edgecolors='black', linewidth=0.5))
    
    legend1 = ax.legend(legend_bubbles, [f'OR = {size}' for size in legend_sizes], 
                       title='Odds Ratio', loc='lower right', frameon=True, fancybox=True, shadow=True)
    legend1.get_title().set_fontsize(10)
    

    # Improve layout
    plt.tight_layout()
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Invert y-axis so most significant terms are at the top
    ax.invert_yaxis()
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Bubble plot saved to: {output_path}")
    else:
        plt.show()
    
    return fig, ax
    

    


# %%

def pathway_enrichment_pipeline(protein_group_df, protgroup_gene_map, binary_var, db_list, missingness_thresh=0.2, imputation_method="mean", fdr_thresh=0.05):

    dep_list = dep_pipeline(protein_group_df, binary_var, missingness_thresh=missingness_thresh, imputation_method=imputation_method, fdr_thresh=0.1)
    deg_list = get_deg_list(dep_list, protgroup_gene_map)

    if len(deg_list) == 0:
        print("No genes identified for enrichment.")
        return None
    else:
        enr = run_enrichr(deg_list, db_list, fdr=fdr_thresh)   
        return enr


# %% [markdown]
# ### main()

# %%

def main(config_path):
    from datetime import datetime
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Required
    protein_group_csv = config["protein_group_csv"]
    clinical_csv = config["clinical_csv"]
    protgroup_gene_map_json = config["protgroup_gene_map_json"]
    var = config["var"]
    mapping = config["mapping"]
    db = config["db"]
    output_dir = config["output_dir"]

    # Optional
    missingness_thresh = config.get("missingness_thresh", 0.2)
    imputation_method = config.get("imputation_method", "mean")
    fdr_thresh = config.get("fdr_thresh", 0.05)

    # Load input files
    protein_group_df = pd.read_csv(protein_group_csv, index_col=0)
    clinical_df = pd.read_csv(clinical_csv, index_col=0)
    with open(protgroup_gene_map_json, 'r') as f:
        protgroup_gene_map = json.load(f)

    print("Starting enrichment pipeline...")
    print("Using clinical variable:", var)

    binary_vars = prepare_categorical_variable_for_pwc(clinical_df, var, fill_value=-1, verbose=False)

    # Create timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_dir = os.path.join(output_dir, timestamp)
    os.makedirs(timestamped_dir, exist_ok=True)

    #pwc_enr_results = {}
    for pwc, binary_var in binary_vars:
        print()
        print(f"Running enrichment for {pwc}...")
        # Run enrichment
        #pthwys = pathway_enrichment_pipeline(
        enr = pathway_enrichment_pipeline(
            protein_group_df,
            protgroup_gene_map,
            binary_var,
            db,
            missingness_thresh=missingness_thresh,
            imputation_method=imputation_method,
            fdr_thresh=fdr_thresh,
        )

        if enr is None or not hasattr(enr, 'results'):
            print(f"No enriched pathways found for {pwc}.")
            continue

        pthwys = enr.results
        #pwc_enr_results[pwc] = pthwys
    

        pwc_dir = os.path.join(timestamped_dir, pwc)
        os.makedirs(pwc_dir, exist_ok=True)

        deg_list = enr.gene_list
        title = f"'{pwc}' gene list (n={len(deg_list)})"
        for db_name in db:
            out_file = os.path.join(pwc_dir, f"{pwc}_{db_name}_bubble_plot.png")
            create_bubble_plot(enr, db_name, x_axis="rich_factor", fdr_thresh=fdr_thresh, title=title, output_path=out_file)

        # Save results
        out_file = os.path.join(pwc_dir, f"{pwc}_enrichment.csv")
        pthwys.to_csv(out_file, index=False)
        print(f"Saved enrichment results to: {out_file}")

    # Save copy of config file
    config_copy_path = os.path.join(timestamped_dir, "config.yaml")
    with open(config_copy_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Saved config copy to: {config_copy_path}")

    return pthwys

# %% [markdown]
# ### execution

# %%
#if __name__ == "__main__":
if __name__ == "__main__" and 'ipykernel' not in sys.modules:  # CLI mode
    parser = argparse.ArgumentParser(description="Pathway Enrichment Pipeline using gseapy")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to config file")
    
    args = parser.parse_args()
    main(args.config_path)

if 'ipykernel' in sys.modules:  # Notebook mode
    config = {
        "protein_group_csv": "protein_group_df.csv",
        "clinical_csv": "clinical_df.csv",
        "protgroup_gene_map_json": "protgroup_gene_map.json",
        "output_dir": "enrichment_results",

        # Clinical variable to test
        "var": "Condition",
        "mapping": {
            '1-C': 0,
            #'2-SC': 1,
            '3-A': 1,
            '4-S': 1
        },

        # Enrichment database
        #"db": "KEGG_2021_Human",
        "db": ["GO_Biological_Process_2021",
               "KEGG_2021_Human"],

        # Optional settings
        "missingness_thresh": 0.2,
        "imputation_method": "mean",
        "fdr_thresh": 0.05
    }

    with open("config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    main("config.yaml")
