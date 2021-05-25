# RECLE-Code

The repository is the debugged code of [RECLE](https://ieeexplore.ieee.org/document/9171289) paper. The main work is to debug the [code](https://github.com/tal-ai/RECLE) provided by the author, and add Chinese annotation for better understanding.
RECLE, the proposed model of this paper, has been successfully run in the local environment.

### Dependency

```
Python >= 3.7
NumPy = 1.16
pandas = 0.24
Tensorflow = 1.13
```

### How to run

```shell
# 1.Git clone the repository
git clone https://github.com/charfole/RECLE-Reproduction.git
cd source

# 2.Change the absolute path of the dataset
# Open the train_RECLE.py file and navigate to lines 12 and 13
# Change the absolute paths of the dataset to your own paths.
train = pd.read_csv('xxx\\RECLE-Reproduction\\code\\data\\fluency_grade_1_2\\train.csv')
validation = pd.read_csv('xxx\\RECLE-Reproduction\\code\\data\\fluency_grade_1_2\\validation.csv')

# 3.Train the RECLE model
python run train_RECLE.py
```

### Results

![da5dd125d96da01451941834264540c](https://charfole-blog.oss-cn-shenzhen.aliyuncs.com/image/da5dd125d96da01451941834264540c-1621607650402.png)

