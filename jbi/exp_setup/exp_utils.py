from pathlib import Path
import optuna as opt
import os


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
