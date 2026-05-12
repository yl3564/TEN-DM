# TEN-DM 
This repo contrains the code for our paper TEN-DM: Topology-Enhanced Diffusion Model for Spatio-Temporal Event Prediction.

## Environment Setup
- Tested OS: Linux
- Python >= 3.7
- PyTorch == 1.7.1
- Tensorboard

## Dependencies:
1. Install PyTorch 1.7.1 with the correct CUDA version.
2. Use the ``pip install -r requirements. txt`` command to install all of the Python modules and packages used in this project.

## Model Training
Data should be one of JP_Earthquake|COVID19|Thefts|311Service|US_Earthquake|.  
Use the following command to train TEN-DM:   
```
python run.py --dataset $dataset
```

Example run command of training TEN-DM on `JP_Earthquake` dataset:  
```
python run.py --dataset JP_Earthquake
```

You can optionally specify additional hyperparameters to control training:
```
python run.py --dataset $dataset --timesteps $timesteps --samplingsteps $samplingsteps --batch_size $batch_size --total_epochs $total_epochs --loss_type $loss_type
```

The trained models are saved in ``ModelSave/``.  
The logs are saved in ``logs/``.  
The test results are saved in ``ModelResult/``.