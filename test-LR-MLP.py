# -*- coding: utf-8 -*-
import numpy as np
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

fr = open('test.txt','r')
lines = fr.readlines()
data_X = []
data_Y = []
for line in lines:
    ls = line.strip('\n').replace('\t',' ').split(' ')
    ls = [float(x) for x in ls]
    data_X.append(ls[0:-1])
    data_Y.append(ls[-1])

#数据标准化
ss_X = StandardScaler().fit(data_X)
X = ss_X.transform(data_X)
#print("数据集X:\n",data_X)
data_Y = np.array(data_Y).reshape(-1,1)
ss_Y = StandardScaler().fit(data_Y)
Y = ss_Y.transform(data_Y)
Y = np.ravel(Y)
#print("数据集Y:\n",data_Y)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)

#LR模型
#设定alpha值及个数
alphas_to_test = np.linspace(0.001,1,num = 100)
#创建模型，保存误差值
lr = linear_model.RidgeCV(alphas=alphas_to_test,store_cv_values=True)
lr.fit(X_train,Y_train)
display("LR截距:",lr.intercept_)  #截距
display("LR系数:",lr.coef_)  #线性模型的系数
score_lr = lr.score(X_test,Y_test)
print("LR回归模型得分:",score_lr)

Y_LR = lr.predict(X_test)
#将预测集Y还原
Y_LR = np.array(Y_LR).reshape(-1,1)
Y_LR_ori = ss_Y.inverse_transform(Y_LR)
Y_LR_ori = np.ravel(Y_LR_ori)
#将测试集Y还原
Y_test = np.array(Y_test).reshape(-1,1)
Y_test_ori = ss_Y.inverse_transform(Y_test)
Y_test_ori = np.ravel(Y_test_ori)

print("Y真实:",Y_test_ori)
print("Y预测_LR:",Y_LR_ori)

#岭系数
print("岭系数:",lr.alpha_)
#loss值
print("loss值:",lr.cv_values_.shape)

#MLP模型
'''
mlp = MLPRegressor(hidden_layer_sizes=(6,2),  activation='relu', solver='adam', \
                   alpha=0.0001, batch_size='auto',learning_rate='constant', \
                   learning_rate_init=0.001, power_t=0.5, max_iter=5000, shuffle=True,\
                   random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, \
                   nesterovs_momentum=True,early_stopping=False,beta_1=0.9, beta_2=0.999, epsilon=1e-08)
'''
mlp = MLPRegressor(solver='lbfgs', alpha=0.0001,hidden_layer_sizes=(4,2), activation='tanh',random_state=1)
mlp.fit(X_train, Y_train)
Y_MLP = mlp.predict(X_test)
score = mlp.score(X_test,Y_test)
print("MLP回归模型得分:",score)

Y_MLP = np.array(Y_MLP).reshape(-1,1)
Y_MLP_ori = ss_Y.inverse_transform(Y_MLP)
Y_MLP_ori = np.ravel(Y_MLP_ori)
print("Y预测_MLP:",Y_MLP_ori)

cengindex = 0
for wi in mlp.coefs_:
    cengindex += 1  # 表示底第几层神经网络。
    print('第%d层网络层:' % cengindex)
    print('权重矩阵维度:',wi.shape)
    print('系数矩阵:\n',wi)

# 画图
# 岭系数跟loss值的关系
plt.plot(alphas_to_test,lr.cv_values_.mean(axis=0)) # axis=0代表方向
# 获取的岭系数值的位置
plt.plot(lr.alpha_,min(lr.cv_values_.mean(axis=0)),'ro')
plt.xlabel('λ')
plt.ylabel('J(β)')
plt.show()
#作图
plt.plot(Y_test_ori,label='real')
plt.plot(Y_MLP_ori,label='MLP')
plt.plot(Y_LR_ori,label='LR')
plt.ylabel('Recovery(%)')
plt.legend()
plt.show()