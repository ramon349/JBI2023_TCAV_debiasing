from pathlib import Path
import optuna as opt
import os
from .exp_helpers.skin_cancer_utils import _skin_transform_params
from .exp_helpers.mammo_utils import _mammo_transform_params

def _get_best_params(log_file):
    optuna_sql = f"sqlite:///{log_file}"
    assert os.path.isfile(log_file)
    study = opt.load_study(study_name="my_study", storage=optuna_sql)
    df = study.trials_dataframe()
    df = df.sort_values(by="value", ascending=True)
    best_trial = df.iloc[0]
    print(f"Best Trial Value: {best_trial['value']}")
    param_d = dict()
    for k, v in best_trial.to_dict().items():
        if k.startswith("params_"):
            new_key = k.replace("params_", "")
            param_d[new_key] = v
    return param_d

def get_argmuents(task): 
    task_d = dict()
    match task: 
        case 'skin':
            task_d["img_col"]= "file"
            task_d["task_col"]= "three_partition_label_cls"
            task_d["mask_col"]= "mask_file"
            task_d['num_task'] = 2
            task_d["demo_col"] ="discrete_fitz",
            task_d['dataset']='ImageDataMask'
            task_d['num_demo'] = 3
            task_d['img_reader'] = 'png'
        case 'mammo':
            task_d["img_col"]= "mayo_dcm"
            task_d["task_col"]= "density_num"
            task_d["demo_col"] ="age_cat",
            task_d['num_task'] = 4
            task_d['num_demo'] = 3
            task_d['dataset'] = 'ImageData'
            task_d['img_reader'] = 'dcm'
    return task_d

def get_transforms(task): 
    match task: 
        case 'skin':
            return  _skin_transform_params()
        case 'mammo': 
            return  _mammo_transform_params()