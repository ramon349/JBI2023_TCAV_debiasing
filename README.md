#Note: 
- Prior code was difficult to use. This branch is being used to update the code for better reproducibility.
- The original code can be found in the "main" branch.
- This branch will be "updated" by default for now.

# Setup Instructions 
- Install the conda environment to ensure the packages are compatible. 
```bash 
 conda create --name jbi --file jbi_env.yml
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python3 -m pip install -r ./reqs.txt
python3 -m pip install -e . 
 ```

# Preparing datasets 
 - I define a class to load Image Date under  datasets/image_data.py 
 - it is able to load png/jpg images supported by PIL 
 - it expects you provide a jbi/config file with the following information 
    - data_path: absolute path to csv file 
    - inside the csf file you should define a split colum for train/val/test 
    - col_info has additional metadata 
        - task_col is the column containing your task labels 
- You can subclass with your loading logic if necessary 
- For debiasing experiments, we require a task variable and a demographic variable 
- The TwoTask datasets will do precisely that by adding a "demo_col" that works the same way as task_Col  

# Training Baseline Model 
-  To train your own baseline mdoel you can use the jbi/config file in "./jbi/config/train_base" 
```bash 
    python3 -m jbi.train --jbi/config_path jbi/config/train_base.json
```
- NOTE: some paths are hardcoded you will need to change those 
- Bellow I will annotate the same jbi/config file with comments off what section does 
    - csv_path : Absolute path to csv file specifying your dataset 
    - num_workers: number of workers used during training 
    - device: a list of gpu devices to use. Provided as a list  for future Distributed Training support 
    - batch_size: Batch size used for training  
    - dataset: which dataset class to be used during training 
        - Single Task Training --> ImageData
        - Multi Task or Debiasing --> TwoTask 
        - TCAV layer experiemnts --> single   (NOTE: More on this later )
        - You can add your own by defining a class and using the DataRegister method
    - Model: Which model to be used  
        - Singe Task --> densenet121 
        - Multi Task --> DensenetTwoTask 
        - Debiasing --> DensenetTwoTaskAdv
        -  Define your own and register using ModelRegister 
    - Trainer: Specifies training algorithm  
        - Single Task -->  ErmTrainer 
        - Multi Task --> TwoTaskTrainer 
        - Adversarial Debiasing --> AdversarialTrainerVanilla  
        - You can add your own by doing the modifications to TrainerRegister 
    - train/test transforms  
        - just a list of transforms. Make sure to maintain the deterministic transforms consistent between train and test 
        - transform_conf then specifies additional parameters
    - col_info: information used by the dataset class 
        "img_col":"file"   is the path to an image file 
        "task_col":"partition_cat"  your task label 
    -  trainer_args: 
        - Training-specific parameters
    - model_parameters : 
        - number of classes in your prediction tasks 
    - "splits":["train", "test", "val"] 
        - the splits we will see during training
    "log_dir": "/home/ramon/jbi_2024/model_logs/"
        -path to training log storage and storage of model weights 
# Training the multi-task model 
    - jbi/config/train_two_task.json
    - follows the same logic as the single task model 
```bash 
    python3 -m jbi.train --jbi/config_path jbi/config/train_two_task.json
```
# TCAV 
```bash 
    python3 -m jbi.train --jbi/config_path jbi/config/tcav_explore.json
```
#  Observing TCAV in model layers 
 - modify jbi/config/tcav_explore.json  to point at your data and your model weights 
 - Run the Following  script: 
 ```bash
 python3 -m jbi.tcav --jbi/config_path ./jbi/config/tcav_explore.json
 ``` 
 - We should get an image simlar to:
 ![image info](jbi/results/figures/TACV_score.png)
 - From those results our "debiasing" target will be 
 - "model.features.denseblock3.denselayer16.conv2"

# Training Debiased model 
    - you will  need to add several parameters to the training 
    - particularly: 
        layer_deibias in trainer_args 
        lambda in trainer_args 
        adv_delay in trainer_args
 ```bash
 python3 -m jbi.train --jbi/config_path ./jbi/config/train_adv_tcav.json 
 ```

# Other Modifications 
The model training code will now output predictions for each dataset split in the log_dir folder specified.
- Hyperparameter tuning was removed but will be added
- So will be the code for generating embeddings
