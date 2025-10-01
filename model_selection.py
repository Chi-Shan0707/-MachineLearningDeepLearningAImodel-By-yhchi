from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris=load_iris()

i_v=iris.data
#iris_value:这些花的的特征值
#v->value
i_t=iris.target
#iris_type:这些花对应的总类
#t->type

# v_train,v_test,t_train,t_test=train_test_split(i_v,i_t,random_state=4,test_size=0.2)
# #把整个集合分成学习的data和测试的两部分

# #random_state:定义随机打乱的初始随机种：这样可以保证每次“随机”分组都可复现
# #test_size=0.2 train:test=8:2
# #这步split操作顺带将数据打乱打乱

# knn=KNeighborsClassifier(n_neighbors=5)
# knn.fit(v_train,t_train)
# Prediction=knn.predict(v_test)
# print(Prediction)
# print(knn.score(v_test,t_test))


from sklearn.model_selection import cross_val_score
knn=KNeighborsClassifier(n_neighbors=5)
scores=cross_val_score(knn,i_v,i_t,cv=10,scoring='accuracy')
#i_v,i_t数据集会自动平均分组
#cv:cross_velidication交叉验证的组数
print(scores)
print(scores.mean())#means:平均数

print("试着去寻找好的参数\n")

for i in range(2,5):
    for j in range(2,5):
        knn=KNeighborsClassifier(n_neighbors=i,p=1)

        scores=cross_val_score(knn,i_v,i_t,cv=j,scoring='accuracy')
        print("n_neighbors:",i,"p:",i,"cv:",j,"scores:",scores.mean())
        print("\n")
'''具体来说，p 参数的含义如下：

# p=1：表示使用曼哈顿距离（Manhattan distance）
# p=2：表示使用欧几里得距离（Euclidean distance），这是默认值
# p=∞：表示使用切比雪夫距离（Chebyshev distance）
# 任意正数p：表示使用闵可夫斯基距离（Minkowski distance）的p次幂
# 在数学上，闵可夫斯基距离定义为：
'''


x_lst=['manhattan','euclidean','chebyshev']
y_lst=[]
for distance_type in x_lst:
    knn=KNeighborsClassifier(n_neighbors=5,metric=distance_type)
    scores=cross_val_score(knn,i_v,i_t,cv=10,scoring='accuracy')
    y_lst.append(scores.mean())

import matplotlib.pyplot as plt
plt.plot(x_lst,y_lst)
plt.xlabel('distance_type')
plt.ylabel('scores')
plt.show()


x_lst=range(1,10)
y_lst=[]

for distance_type in x_lst:
    knn=KNeighborsClassifier(n_neighbors=9,p=distance_type)
#    scores=cross_val_score(knn,i_v,i_t,cv=10,scoring='accuracy')
    loss=-cross_val_score(knn,i_v,i_t,cv=10,scoring='neg_mean_squared_error')
    y_lst.append(loss.mean())

plt.plot(x_lst,y_lst)
plt.xlabel('distance_type')
plt.ylabel('loss')
plt.show()

#p参数和metric参数都决定了“距离”种类