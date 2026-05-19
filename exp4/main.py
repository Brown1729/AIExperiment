'''
在环境中，使用如下命令安装libsvm:
    pip install -U libsvm-official
'''
# 导入 libsvm：
from libsvm.svmutil import *
from libsvm.svm import *

'''
分别载入训练集，测试集。验证集，y代表种类，x代表特征。
例如在DNA样本中,y有1，2，3共3种取值。
'''
print("正在加载数据...")
y, x = svm_read_problem('../dna/dna.scale.tr')     # 训练
yv, xv = svm_read_problem('../dna/dna.scale.val')  # 验证
yt, xt = svm_read_problem('../dna/dna.scale.t')      # 测试
prob = svm_problem(y, x)
'''
参数解释：
-t: 核函数类型，默认为2
    0--线性核
    1--多项式核
    2--高斯核
    3--sigmoid核
    4--预计算核
-c: 调整SVM中cost参数，默认为1
-g: 调整核函数的gamma参数，默认为1/num_features
'''
param = svm_parameter(
    '-t 2 -c 2 -g 0.0078125'
)
'''
c 、 g两个参数影响分类准确率，可以通过网格搜索得到最佳参数
原理是以c为行，以g为列，设定格点步长，遍历每一个格点，使用验证集测试出分类性能最佳的一组值。
但是寻优过程非常漫长，耗时取决于c,g参数选取范围，以及网格精度。

这里给出一组表现较好的参数
'-t 0 -c 0.03125 -g 0.0078125'
'-t 1 -c 16 -g 1'
'-t 2 -c 2 -g 0.0078125'
'''
model = svm_train(prob, param)
svm_save_model('model_svm', model)
print('Train:')
p_label, p_acc, p_val = svm_predict(y, x, model)
# print("p_label: ", p_label)
print("p_acc: ", p_acc)
# print("p_val: ", p_val)
print('Test:')
p_label, p_acc, p_val = svm_predict(yt, xt, model)
