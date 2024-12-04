import subprocess
from sklearn.model_selection import ParameterGrid

if __name__ == '__main__':
    param_grid = {
        '1_model_type': ['UNet'],
        '2_learning_rate': [1e-03],
        '3_weight_decay': [0],
        '5_time': [0],
    }

    param_combinations = ParameterGrid(param_grid)

    count = 1

    for params in param_combinations:
        # Usage: python <Script Name> <GPU ID> <Model Type> <Learning Rate> <Weight Decay> <Fold ID> <Time> <Train/Test Flag>

        subprocess.call(["python", "/data1/users/liliang/zwbb/UNet_322_10.py", '0', str(params['1_model_type']), str(params['2_learning_rate']), str(params['3_weight_decay']), str(params['5_time']), 'Train'])

        count += 1