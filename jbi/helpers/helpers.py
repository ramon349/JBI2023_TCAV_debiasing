import torch 
import pandas as pd 
import numpy as np 

def make_mult_col(df, col_name, is_pred=False):
    if is_pred:
        arr = torch.tensor(np.vstack(df[col_name]))
        arr = torch.nn.functional.softmax(arr, dim=1).numpy()
    else:
        arr = np.vstack(df[col_name])
    if arr.shape[1] != 1:
        for e in range(arr.shape[1]):
            df[f"{col_name}_{e}"] = arr[:, e]
        del df[col_name]
    else:
        df[col_name] = arr    
def proc_inference(ground_truths, preds):
    gt_df = pd.DataFrame(ground_truths)
    pred_df = pd.DataFrame(preds)
    for e in list(gt_df.keys()):
        if e.endswith("t"):
            make_mult_col(gt_df, e)
    for e in list(pred_df.keys()):
        if e.endswith("p"):
            make_mult_col(pred_df, e, is_pred=True)
    # ret_df = pd.merge(gt_df,pred_df,on='path')
    ret_df = pd.concat([gt_df, pred_df], axis=1)
    return ret_df


def process_df(df, col_indicator="_"):
    for e in list(df.keys()):
        if col_indicator in e:
            make_mult_col(df, e)