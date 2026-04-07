"""
Forecasting Asset Returns with Sentiment-Enhanced FIGARCH Models
Author: Mostafa Abdolahi Moghadam

Description:
This script fits Fractionally Integrated GARCH (FIGARCH) models on financial return 
data, optionally augmented with exogenous variables (machine-learning impact probabilities 
and FinBERT sentiment scores). It evaluates 8 different exogenous feature sets across 
four distinct intraday return time frames (OC, OO, CC, CO). The script outputs 
goodness-of-fit metrics (AIC, BIC, LLF) and summaries of coefficient statistical significance.
"""

import os
import glob
import random
import numpy as np
import pandas as pd
from arch import arch_model
from pandas import ExcelWriter
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

# Set random seeds for reproducibility
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Define directories using relative paths for repository consistency
data_dir = "./datasets/Alpha_FinBERT_datasets"
output_dir = "./results"
os.makedirs(output_dir, exist_ok=True)

# Find all generated dataset files from the Bag-of-Words script
csv_files = glob.glob(os.path.join(data_dir, "RF_BOW_*_dataset.csv"))

logreturn_types = ['logreturn_OC', 'logreturn_CC', 'logreturn_OO', 'logreturn_CO']

# Define the 8 exogenous variable sets as described in the paper methodology
exog_sets = {
    'No_Exog': [],
    
    'Exog_Set_1': ['RFIP_Pos_OC', 'RFIP_Neg_OC'],
    
    'Exog_Set_2': ['RFIP_Pos_OO', 'RFIP_Neg_OO'],
    
    'Exog_Set_3': ['RFIP_Pos_CC', 'RFIP_Neg_CC'],
    
    'Exog_Set_4': ['RFIP_Pos_CO', 'RFIP_Neg_CO'],
    
    'Exog_Set_5': ['RFIP_Pos_OC', 'RFIP_Pos_OO', 'RFIP_Pos_CC', 'RFIP_Pos_CO',
                   'RFIP_Neg_OC', 'RFIP_Neg_OO', 'RFIP_Neg_CC', 'RFIP_Neg_CO'],
    
    'Exog_Set_6': ['average_score_positive', 'average_score_neutral', 'average_score_negative'],
    
    'Exog_Set_7': ['RFIP_Pos_OC', 'RFIP_Neg_OC', 'RFIP_Pos_OO', 'RFIP_Neg_OO', 
                   'average_score_positive', 'average_score_neutral', 'average_score_negative'],
    
    'Exog_Set_8': ['average_score_positive', 'average_score_neutral', 'average_score_negative',
                   'RFIP_Pos_OC', 'RFIP_Pos_OO', 'RFIP_Pos_CC', 'RFIP_Pos_CO',
                   'RFIP_Neg_OC', 'RFIP_Neg_OO', 'RFIP_Neg_CC', 'RFIP_Neg_CO']
}

results = []
scaler = StandardScaler()

# Iterate over each asset dataset
for csv_file in csv_files:
    filename = os.path.basename(csv_file)
    ticker = filename.split("_")[2]

    df = pd.read_csv(csv_file)
    
    # Handle specific commodity naming convention
    if ticker == "commodity":
        df.rename(columns={"logreturn": "logreturn_OO"}, inplace=True)
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    # Iterate over the four return time horizons
    for lr_col in logreturn_types:
        if lr_col not in df.columns: 
            print(f"Skipping {lr_col} for {ticker}: column missing.")
            continue
        
        df_clean = df.dropna(subset=[lr_col])
        y = df_clean[lr_col].copy()

        # Fit model for each exogenous variable set
        for exog_name, exog_vars in exog_sets.items():
            try:
                if exog_vars:
                    # Filter to only variables available in the current dataframe
                    available_vars = [var for var in exog_vars if var in df_clean.columns]
                    if not available_vars:
                        continue
                        
                    exog = df_clean[available_vars].copy().fillna(0)
                    aligned_df = pd.concat([y, exog], axis=1).dropna()
                    
                    y_aligned = aligned_df[lr_col]
                    exog_aligned = aligned_df[available_vars]

                    # Scale exogenous variables for numerical stability
                    exog_scaled = pd.DataFrame(
                        scaler.fit_transform(exog_aligned),
                        columns=exog_aligned.columns,
                        index=exog_aligned.index
                    )
                    
                    if exog_name != 'No_Exog':
                        constant_cols = [c for c in exog_scaled.columns if exog_scaled[c].nunique() == 1]
                        exog_scaled_clean = exog_scaled.drop(columns=constant_cols)
                    
                else:
                    aligned_df = y.dropna().to_frame()
                    y_aligned = aligned_df[lr_col]
                    exog_scaled_clean = None

                # Initialize and fit the FIGARCH model
                model = arch_model(
                    y=y_aligned,
                    vol='FIGARCH',
                    p=1,
                    q=1,
                    mean='ARX' if exog_scaled_clean is not None else 'Constant',
                    x=exog_scaled_clean,
                    dist='t',
                    lags=1
                )
                
                res = model.fit(disp='off', cov_type='robust')
                
                try:
                    pvals = res.pvalues.to_dict()
                except Exception as e:
                    print(f"P-value extraction error for {ticker}: {e}")
                    pvals = "Singular Matrix"

                # Store metrics
                results.append({
                    'Asset': ticker,
                    'LogReturn': lr_col,
                    'Exog_Set': exog_name,
                    'AIC': res.aic,
                    'BIC': res.bic,
                    'LLF': res.loglikelihood,
                    'Params': res.params.to_dict(),
                    'P-values': pvals
                })

                print(f"Successfully fitted {ticker} | {lr_col} | {exog_name}")

            except Exception as e:
                print(f"Failed to fit {ticker} | {lr_col} | {exog_name}: {e}")

results_df = pd.DataFrame(results)

# ==============================================================================
# COUNTING SIGNIFICANT VARIABLES
# ==============================================================================

# Initialize dictionaries to count appearances and significance levels
exog_variable_counts = defaultdict(lambda: defaultdict(int))
exog_significant_10  = defaultdict(lambda: defaultdict(int))
exog_significant_5   = defaultdict(lambda: defaultdict(int))
exog_significant_1   = defaultdict(lambda: defaultdict(int))

# Constraint: Prevent look-ahead bias by restricting certain exogenous sets 
# for Open-to-Close and Open-to-Open return horizons.
A1 = results_df[
    (results_df['LogReturn'].isin(['logreturn_OC', 'logreturn_OO'])) &
    (results_df['Exog_Set'].isin(['Exog_Set_1', 'Exog_Set_2', 'Exog_Set_6', 'Exog_Set_7', 'No_Exog']))
][['Exog_Set', 'P-values']]

# No constraints for Close-to-Close and Close-to-Open horizons
A2 = results_df[results_df['LogReturn'].isin(['logreturn_CC', 'logreturn_CO'])][['Exog_Set', 'P-values']]

# Aggregate counts
for _, row in pd.concat([A1, A2]).iterrows():
    exog_set = row['Exog_Set']
    pvals    = row['P-values'] 
    if isinstance(pvals, dict):
        for var, pval in pvals.items():
            exog_variable_counts[exog_set][var] += 1
            if pval < 0.10:
                exog_significant_10[exog_set][var] += 1
            if pval < 0.05:
                exog_significant_5[exog_set][var] += 1
            if pval < 0.01:
                exog_significant_1[exog_set][var] += 1
                
# Combine into a summary DataFrame indexed by (Exog_Set, Variable)
summary_records = []
for exog_set in sorted(exog_variable_counts):
    for var in exog_variable_counts[exog_set]:
        summary_records.append({
            'Exog_Set':         exog_set,
            'Variable':         var,
            'Count':            exog_variable_counts[exog_set][var],
            'Significant @10%': exog_significant_10[exog_set].get(var, 0),
            'Significant @5%':  exog_significant_5[exog_set].get(var, 0),
            'Significant @1%':  exog_significant_1[exog_set].get(var, 0),
        })

summary_df = pd.DataFrame(summary_records).set_index(['Exog_Set', 'Variable']).fillna(0).astype(int)

# ==============================================================================
# EXPORT RESULTS
# ==============================================================================

output_file = os.path.join(output_dir, "FIGARCH_Comparative_Results.xlsx")

with ExcelWriter(output_file, engine='xlsxwriter') as writer:
    # Save individual log return results to separate sheets
    for lr_col in logreturn_types:
        subset = results_df[results_df['LogReturn'] == lr_col]
        if not subset.empty:
            subset.to_excel(writer, sheet_name=lr_col[:31], index=False)
            
    # Save the consolidated p-value significance counts
    summary_df.to_excel(writer, sheet_name="P-values Summary", index=True)

print(f"\nAll modeling complete. Results successfully saved to {output_file}")
