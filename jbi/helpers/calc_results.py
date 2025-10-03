from sklearn.metrics import roc_auc_score
import pandas as pd
from glob import glob
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
)
from sklearn.metrics import roc_auc_score
import pandas as pd
import os
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from scipy.stats import ttest_ind

def gen_key_name(row, metric):
    return f"{row['model']}_all_{metric}_{row['group']}"
def define_asterisk(p_val):
    if p_val <= (1e-4):
        return "****"
    if p_val <= 0.0001:
        return "****"
    if p_val <= 0.001:
        return "**"
    if p_val <= 0.05:
        return "*"
    else:
        return ""

def bootstrap_res(all_df, num_classes, seed=42, min_p=50, max_p=75, class_map=None):
    """ """
    np.random.seed(seed)
    class_results = {"metric": list(), "val": list(), "class": list()}
    for it in tqdm(range(100), total=100):
        rand_samp = np.random.randint(min_p, max_p) / 100
        sub_df = all_df.sample(frac=rand_samp, random_state=seed + it)
        for i in range(num_classes):
            binary_metrics = {
                "precision": precision_score,
                "recall": recall_score,
                "f1_score": f1_score,
            }
            cont_metrics = {"auc": roc_auc_score}
            for k, metric_func in binary_metrics.items():
                result_list = metric_func(sub_df["class"] == i, sub_df["max_pred"] == i)
                name = class_map[i]
                class_results["val"].append(result_list)
                class_results["metric"].append(k)
                class_results["class"].append(name)
            for k, metric_func in cont_metrics.items():
                name = class_map[i]
                try:
                    res = roc_auc_score(sub_df["class"] == i, sub_df[f"task_p_{i}"])
                except:
                    res = 0.5
                class_results["val"].append(res)
                class_results["metric"].append(k)
                class_results["class"].append(name)
            if num_classes > 2:
                res = balanced_accuracy_score(sub_df["class"], sub_df["max_pred"])
                class_results["val"].append(res)
                class_results["metric"].append("balanced_acc")
                class_results["class"].append(name)
    return pd.DataFrame(class_results)

def make_significance_map_classy(raw_boots, ref_mode="Baseline_None_None"):
    ref_map = dict()
    for sub_n, sub_d in raw_boots.groupby(by=[ "metric", "class"]):
        ref_model = sub_d[sub_d["model_name"] == ref_mode].copy()
        metric = sub_n[0]
        group = sub_n[1]
        for name, sub_exp in sub_d.groupby(by="model_name"):
            pval = ttest_ind(sub_exp["val"], ref_model["val"]).pvalue
            asterisk = define_asterisk(pval)
            ref_key = f"{name}_{metric}_{group}"
            ref_map[ref_key] = asterisk
    return ref_map

def interval_calc(df, cols, rows, num_sigs):
    # cols = ['auc','precision','recall','f1_score']
    # rows =  df['site'].unique()
    new_df = {e: list() for e in cols}
    new_df["group"] = list()
    for row in rows:
        site_df = df[df["class"] == row].copy()
        for col in cols:
            site_metric = site_df[site_df["metric"] == col].copy()
            mu = site_metric["val"].mean()
            mu_l, mu_h = np.percentile(site_metric["val"], [2.5, 97.5])
            new_df[col].append(
                f"{mu:0.{num_sigs}f} ({mu_l:0.{num_sigs}f},{mu_h:0.{num_sigs}f})"
            )
        new_df["group"].append(row)
    return pd.DataFrame(new_df)


def gen_results(p_df,gt_df,model_name): 
    class_map = {0:"non-neoplastic", 2:"malignant", 1:"benign"}
    all_df = pd.merge(gt_df,p_df,left_on='file',right_on='paths')
    all_df['class'] = all_df['three_partition_label_cls']
    all_df['max_pred'] = all_df[[e for e in all_df if e.startswith('task_p_')]].values.argmax(axis=1)
    boot_df = bootstrap_res(all_df,num_classes=3,seed=42,class_map=class_map)
    boot_df['model_name']=model_name 
    int_df = interval_calc(boot_df,cols=['auc','precision','recall','f1_score','balanced_acc'],rows=['non-neoplastic','benign','malignant'],num_sigs=3)
    int_df['model'] = model_name
    return int_df 
