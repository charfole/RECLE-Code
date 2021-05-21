import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
import tensorflow as tf
import os
import copy

def compute_distance(matrix, v, distance_metric):
    if(distance_metric=='cosine'):
        return cosine_similarity(matrix, v.reshape(1, -1)).reshape(-1)
    elif(distance_metric=='l2'):
        return euclidean_distances(matrix, v.reshape(1, -1)).reshape(-1)
    elif(distance_metric=='l1'):
        return manhattan_distances(matrix, v.reshape(1, -1)).reshape(-1)
    else:
        raise ValueError('Invalid distance metric, must be in [cosine, l1, l2]')

#compute cosine distance between two vectors
def cos_sim(a, b):
    normalize_a = tf.nn.l2_normalize(a,1)        
    normalize_b = tf.nn.l2_normalize(b,1) 
    cos_similarity = tf.reduce_sum(tf.multiply(normalize_a,normalize_b), axis=1)
    return cos_similarity

    
def prepareInput(data, groupSize=int(4e5), alpha=None, beta=None, infer_weight=True, polarity='positive'):
    if(infer_weight):
        data = inferWeight(data, alpha, beta) # 加入贝叶斯置信度估计，改变数据中的第二列
    groups, weights = createGroupsRandom(data, groupSize, polarity)
    return groups, weights


def inferWeight(data, alpha=None, beta=None):    
    '''
    Args:
    alpha and beta are beta distribution parameters.
    It is suggested that alpha/(alpha+beta) = ratio of positive examples in training data,
    and sum of alpha and beta should be equal to max vote number

    for example, if 40% of the training data is positive examples, and max vote is 5, then alpha=2, beta=3 as suggested
    '''
    votes = data[:,1]   # 获取每个样本认为该样本为正的众包工人的数量
    maxVote = max(votes) # 找到该数据中的最多众包工人数
    weights = []
    for i in range(votes.shape[0]):
        v = votes[i]
        if(v>=(1+maxVote)/2):   #大部分人认为这是正样本
            if(alpha==None or beta==None):
                weights.append(float(v/maxVote))
            else:
                weights.append(float((v+alpha)/(maxVote+alpha+beta)))
                # 对应着论文4.2处的公式，为解决每个样本标注人数不一致的问题所加入的贝叶斯置信估计项
        else:                   #大部分人认为这是负样本
            if(alpha==None or beta==None):
                weights.append(1-float(v/maxVote))
                # 少于一半，说明该样本是负的，因此要计算标注为负的比率
                # maxvote - v即代表标注为负的数量
            else:
                weights.append(float((maxVote-v+alpha)/(maxVote+alpha+beta)))
    new_data = copy.deepcopy(data)
    new_data[:,1] = weights
    return new_data # 将beta分布处理后的正负标注比率替代掉原来的第二列，即替代了认为正样本的人数

def splitFeatureWeight(x):
    return x[:,1:], x[:,0] # 分别返回特征列和贝叶斯权重列

def createGroupsRandom(data, groupSize=int(1e6), polarity='positive'):
    # query即代表基准组，如果query为正，即让正样本之间距离最小化，正负样本距离最大化，反之亦然
    if(polarity=='positive'): # use positive example as query 
        print('use positive example as query ')
        positive = data[np.where(data[:,0]==1)] #1为正，0为负
        negative = data[np.where(data[:,0]==0)]
    elif(polarity=='negative'): # use negative example as query
        print('use negative example as query')
        positive = data[np.where(data[:,0]==0)] #1为负，0为正
        negative = data[np.where(data[:,0]==1)]
    else:
        raise ValueError('polarity must be positive or negative')
    posNum = positive.shape[0]
    negNum = negative.shape[0]

    if(posNum==0 or negNum==0):
        raise ValueError('There are no positive or negative examples in data.')

    # 正负样本的小组由十万行样本shuffle后组合而成
    idx = np.random.randint(low=0, high=posNum, size=groupSize)
    query = np.array([positive[i,1:] for i in idx]) # 从第1列开始取，第0列就不要了
    posDoc = shuffle(query)
    idx = np.random.randint(low=0, high=negNum, size=groupSize)
    negDoc0 = np.array([negative[i,1:] for i in idx])
    negDoc1 = shuffle(negDoc0)
    negDoc2 = shuffle(negDoc0)
    # 创建三组负样本
    
    query, _ = splitFeatureWeight(query)
    posDoc, posDocW = splitFeatureWeight(posDoc)
    negDoc0, negDoc0W = splitFeatureWeight(negDoc0)
    negDoc1, negDoc1W = splitFeatureWeight(negDoc1)
    negDoc2, negDoc2W = splitFeatureWeight(negDoc2) # 权重(xxx_w变量)即为解决不一致性所引入的贝叶斯置信估计
    
    groups = (query, posDoc, negDoc0, negDoc1, negDoc2)
    weights = (posDocW, negDoc0W, negDoc1W, negDoc2W)
    return groups, weights