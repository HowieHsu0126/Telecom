import datetime
import pandas as pd
from automl import AutoML
from data import Datasets
from model import Models
from utils import Logger


def main(project_name='baseline'):
    logger = Logger.setup_logger(project_name)

    # 数据文件路径
    file_paths = {
        'train': '/home/hwxu/Projects/Competition/Telecom/Input/processed/train.csv',
        'val': '/home/hwxu/Projects/Competition/Telecom/Input/processed/val.csv',
    }

    output_path = f'/home/hwxu/Projects/Competition/Telecom/Output/submissions/prediction_{datetime.datetime.now()}.csv'

    # 数据
    train = pd.read_csv(file_paths['train'])
    X = train.drop(columns=['msisdn'])
    y = train['msisdn']

    X_val = pd.read_csv(file_paths['val'])
    
    print(X.shape, y.shape, X_val.shape)
    # 模型
    model_controller = Models()
    model_controller.run_pipeline(
        X, y, X_val, output_path, logger,
        tune_hyperparameters=False,
        adversarial_val=False,
        use_pseudo_labeling=False,
    )

    # model = AutoML.run_automl(X, y, logger)
    # AutoML.predict_and_save(model, X_val, validation_res, output_path, logger)


if __name__ == "__main__":
    main('MLP_Mode')
