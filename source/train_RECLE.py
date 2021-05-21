import warnings
warnings.filterwarnings('ignore')
from RECLE import RECLE
from utils import *
import numpy as np
import pandas as pd
import os


# first column is ground truth label, second column is crowdsourced votes (set to 1 if you have no crowdsourced labels)
# 注意：这里的训练集和验证集路径需要进行修改，最好用绝对路径
train = pd.read_csv('D:\\CST\\ReserchAndCompetition\\Research\\NLP\\RECLE\\code\\data\\fluency_grade_1_2\\train.csv')
validation = pd.read_csv('D:\\CST\\ReserchAndCompetition\\Research\\NLP\\RECLE\\code\\data\\fluency_grade_1_2\\validation.csv')
train = np.array(train)
validation = np.array(validation)

dimension = train.shape[1]-2    # 特征数,减2是因为第一列为label，第二列为该样本打标签的人数
gamma = 10.0                    
max_iter = 500

# 注意：这里的模型保存路径需要进行修改，最好用绝对路径
save_path = 'D:\\CST\\ReserchAndCompetition\\Research\\NLP\\RECLE\\fluency_grade_1_2\\RECLE\\model'
summaries_dir = 'D:\\CST\\ReserchAndCompetition\\Research\\NLP\\RECLE\\fluency_grade_1_2\\RECLE\\summary'

if(not os.path.exists(save_path)):
    os.makedirs(save_path)
if(not os.path.exists(summaries_dir)):
    os.makedirs(summaries_dir)

# grid search for parameters
# 一共384种参数组合，即需要对384个模型进行训练
layer_1_list = [64, 128]    # l1_n: number of neurons in the first layer
layer_2_list = [32, 64]     # l2_n: number of neurons in the second layer
dropout_rate_list = [0.5, 0.3]   # dropout的比率
reg_scale_list = [2.0, 5.0]      # 正则化惩罚系数
lr_rate_list = [0.05]            # 逻辑回归的学习率
batchSize_list = [256, 512]      # batchSize大小
activation_list = ['tanh', 'sigmoid']    # 激活函数
distance_metric_list = ['cosine', 'l1', 'l2']    # 距离评判准则
space_list = ['raw', 'embedding']    # 向量的空间
done_list = [x for x in os.listdir(save_path)]   # 存储已经训练完毕的模型名称

for dropout_rate in dropout_rate_list:
    for l1_n in layer_1_list:
        for l2_n in layer_2_list:
            for reg_scale in reg_scale_list:
                for batchSize in batchSize_list:
                    for lr_rate in lr_rate_list:
                        for activation in activation_list:
                            for distance_metric in distance_metric_list:
                                for space in space_list:
                                    try:
                                        model_name = 'RECLE_l1_{}_l2_{}_lr_{}_penalty_{}_bs_{}_dropout_{}_activation_{}_metric_{}_space_{}'.format(\
                                            l1_n, l2_n, lr_rate, reg_scale, batchSize, dropout_rate, activation, distance_metric, space)

                                        if(l1_n<=l2_n or model_name in done_list):
                                            print('pass model {}'.format(model_name))
                                            continue

                                        model = RECLE(dimension, l1_n, l2_n, gamma, distance_metric, train, validation)
                                        model.buildRLL(lr_rate, reg_scale, dropout_rate, activation)
                                        model.train_and_evaluate(train, validation, batchSize, 
                                                                max_iter, activation, save_path, summaries_dir, 
                                                                model_name, space=space, use_base_group=False)
                                    except Exception as e:
                                        print(e)