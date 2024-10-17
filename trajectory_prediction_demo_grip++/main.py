import glob
import os 
import torch
from preprocess import generate_data
from train import run_trainval
from model.model import Model
from evaluate import evaluate_model
from visualize import visulization
from test import run_test

# TODO: 日志保存问题妥善处理

batch_size_train = 64 
batch_size_val = 32


data_path = 'data'
model_path = 'result/trained_models/model_epoch_0049.pt'
processd_data_path = 'result/processed_files'
visualizepred_path = 'result/visulize_val'
testresult_path = 'result/test_result'

trainval_data_path=os.path.join(processd_data_path, 'train_val_data.pkl')
test_data_path=os.path.join(processd_data_path, 'test_data.pkl')

# 数据与预处理
train_file_path_list = sorted(glob.glob(os.path.join(data_path, 'prediction_train/*.txt')))
test_file_path_list = sorted(glob.glob(os.path.join(data_path, 'prediction_test/*.txt')))

print('Generating Training Data.')
generate_data(train_file_path_list, pra_is_train=True)

print('Generating Testing Data.')
generate_data(test_file_path_list, pra_is_train=False)


# 训练模型
graph_args={'max_hop':2, 'num_node':120}
model = Model(in_channels=4, graph_args=graph_args, edge_importance_weighting=True)
dev = torch.device("cpu")
model.to(dev)
run_trainval(model, trainval_data_path)


# 数据评估指标计算
evaluate_model(model_path,trainval_data_path)


# 可视化
visulization(model_path,trainval_data_path,indx=10,visualize_path=visualizepred_path)


# 测试数据
run_test(model_path, test_data_path,testresult_path)