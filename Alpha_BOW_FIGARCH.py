#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 31 23:04:08 2025

@author: mostafamoghadam
"""

import os
import glob
import pandas as pd
import os, random, numpy as np
from arch import arch_model
from pandas import ExcelWriter
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)



script_dir = os.path.dirname(os.path.abspath("Alpha_BOW_FIGARCH.py"))
csv_files = glob.glob(os.path.join(script_dir, "RF_BOW_*_dataset.csv"))

logreturn_types = ['logreturn_OC', 'logreturn_CC', 'logreturn_OO', 'logreturn_CO']

exog_sets = {
    'No_Exog': [],
    
    'Exog_Set_1': ['RFIP_Pos_OC', 'RFIP_Neg_OC'],
    
    'Exog_Set_2': ['RFIP_Pos_OO', 'RFIP_Neg_OO'],
    
    'Exog_Set_3': ['RFIP_Pos_CC', 'RFIP_Neg_CC'],
    
    'Exog_Set_4': ['RFIP_Pos_CO', 'RFIP_Neg_CO'],
    
    'Exog_Set_5': ['RFIP_Pos_OC', 'RFIP_Pos_OO', 'RFIP_Pos_CC', 'RFIP_Pos_CO',
                   'RFIP_Neg_OC', 'RFIP_Neg_OO', 'RFIP_Neg_CC', 'RFIP_Neg_CO'],
    
    
    'Exog_Set_6': ['average_score_positive', 'average_score_neutral', 'average_score_negative',
                   'average_combined_score'],
    
    'Exog_Set_7': ['RFIP_Pos_OC', 'RFIP_Neg_OC', 'RFIP_Pos_OO', 'RFIP_Neg_OO', 'average_score_positive', 'average_score_neutral', 
                   'average_score_negative', 'average_combined_score'],
    
    'Exog_Set_8': ['average_score_positive', 'average_score_neutral', 'average_score_negative',
                   'average_combined_score','RFIP_Pos_OC', 'RFIP_Pos_OO', 'RFIP_Pos_CC', 'RFIP_Pos_CO',
                   'RFIP_Neg_OC', 'RFIP_Neg_OO', 'RFIP_Neg_CC', 'RFIP_Neg_CO']
}

results = []
scaler = StandardScaler()

for csv_file in csv_files:
    filename = os.path.basename(csv_file)
    ticker = filename.split("_")[2]

    df = pd.read_csv(csv_file)
    
    if ticker == "commodity":
        df.rename(columns={"logreturn": "logreturn_OO"}, inplace=True)
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    for lr_col in logreturn_types:
        if lr_col not in df.columns: 
            print(f"Skipping {lr_col} for {ticker}: column missing.")
            continue
        
       # df2 = df.dropna(subset=[lr_col], inplace=False)
        df_clean = df.dropna(subset=lr_col)
        y = df_clean[lr_col].copy()

        for exog_name, exog_vars in exog_sets.items():
            try:
                if exog_vars:
                    exog = df_clean[exog_vars].copy().fillna(0)
                    aligned_df = pd.concat([y, exog], axis=1).dropna()
                    y_aligned = aligned_df[lr_col]
                    exog_aligned = aligned_df[exog_vars]

                    exog_scaled = pd.DataFrame(
                        scaler.fit_transform(exog_aligned),
                        columns=exog_aligned.columns,
                        index=exog_aligned.index
                    )
                    
                    if exog_name != 'No_Exog':
                        constant_cols = [c for c in exog_scaled.columns if exog_scaled[c].nunique() == 1]
                        print("Constant columns: ", constant_cols)
                        exog_scaled_clean = exog_scaled.drop(columns=constant_cols)
                    
                else:
                    aligned_df = y.dropna().to_frame()
                    y_aligned = aligned_df[lr_col]
                    exog_scaled_clean = None

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
                res = model.fit(disp='off',cov_type='robust')
                
                try:
                    pvals = res.pvalues.to_dict()
                except Exception as e:
                    print(e)
                    pvals = "Singular Matrix"

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

                print(f"Fitted {ticker} {lr_col} with {exog_name}")

            except Exception as e:
                print(f"Failed to fit {ticker} {lr_col} with {exog_name}: {e}")

results_df = pd.DataFrame(results)


###COUNTING SIGNIFICANT VARIABLES

""" The former version which was not counting per Exog_Set
# Count appearances and significance
variable_counts = defaultdict(int)
significant_10 = defaultdict(int)
significant_5 = defaultdict(int)
significant_1 = defaultdict(int)


A1= results_df[['Exog_Set', 'P-values']][
    (results_df['LogReturn'].isin(['logreturn_OC', 'logreturn_OO'])) &
    (results_df['Exog_Set'].isin(['Exog_Set_1', 'Exog_Set_2', 'Exog_Set_6', 'Exog_Set_7', 'No_Exog']))]

A2= results_df[['Exog_Set', 'P-values']][results_df['LogReturn'].isin(['logreturn_CC', 'logreturn_CO'])]

for row in pd.concat([A1, A2]):
    if isinstance(row, dict):
        for var, pval in row.items():
            variable_counts[var] += 1
            if pval < 0.10:
                significant_10[var] += 1
            if pval < 0.05:
                significant_5[var] += 1
            if pval < 0.01:
                significant_1[var] += 1


# Combine into a summary DataFrame
summary_df = pd.DataFrame({
    'Count': variable_counts,
    'Significant @10%': significant_10,
    'Significant @5%': significant_5,
    'Significant @1%': significant_1
}).fillna(0).astype(int)
"""
###########################
""" ## THIS IS FOR THE TIME WE USE THE EXISTING RESULTS EXCEL FILE
import math
def safe_parse_pvalues(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return eval(x, {"nan": float('nan'), "inf": float('inf'), "__builtins__": {}})
        except Exception:
            return x  # return as-is if unparseable
    return x
# Replace the previous apply line with:
results_df['P-values'] = results_df['P-values'].apply(safe_parse_pvalues)
"""

# Count appearances and significance per exogenous variable set
exog_variable_counts = defaultdict(lambda: defaultdict(int))
exog_significant_10  = defaultdict(lambda: defaultdict(int))
exog_significant_5   = defaultdict(lambda: defaultdict(int))
exog_significant_1   = defaultdict(lambda: defaultdict(int))
A1 = results_df[
    (results_df['LogReturn'].isin(['logreturn_OC', 'logreturn_OO'])) &
    (results_df['Exog_Set'].isin(['Exog_Set_1', 'Exog_Set_2', 'Exog_Set_6', 'Exog_Set_7', 'No_Exog']))
][['Exog_Set', 'P-values']]

A2 = results_df[results_df['LogReturn'].isin(['logreturn_CC', 'logreturn_CO'])][['Exog_Set', 'P-values']]

"""THIS IS FOR THE TIME WE USE THE EXISTING RESULTS EXCEL FILE
for _, row in pd.concat([A1, A2]).iterrows():
    exog_set = row['Exog_Set']
    pvals    = row['P-values']
    if isinstance(pvals, dict):
        for var, pval in pvals.items():
            try:
                if math.isnan(pval):   # skip nan p-values
                    continue
            except TypeError:
                continue
            exog_variable_counts[exog_set][var] += 1
            if pval < 0.10:
                exog_significant_10[exog_set][var] += 1
            if pval < 0.05:
                exog_significant_5[exog_set][var] += 1
            if pval < 0.01:
                exog_significant_1[exog_set][var] += 1

"""

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


output_file = os.path.join(script_dir, "FIGARCH_Comparative_Results.xlsx")
with ExcelWriter(output_file, engine='xlsxwriter') as writer:
    for lr_col in logreturn_types:
        subset = results_df[results_df['LogReturn'] == lr_col]
        if not subset.empty:
            subset.to_excel(writer, sheet_name=lr_col[:31], index=False)
    summary_df.to_excel(writer, sheet_name="P-values Summary", index = True)
print(f"Results saved to {output_file}")



'''
for n, i in enumerate(csv_files):
    ticker = os.path.basename(i).split("_")[2]
    print(n,"========>",ticker)
'''












