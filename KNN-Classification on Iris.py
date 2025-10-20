import numpy
from sklearn import datasets
from sklearn.model_selection import train_test_split
#引入分割数据集工具
from sklearn.neighbors import KNeighborsClassifier
#k近邻分类

# KNN k个最近的邻居

iris=datasets.load_iris()
#加载irisのdatabase


    

i_v=iris.data
#iris_value:这些花的的特征值
#v->value
i_t=iris.target
#iris_type:这些花对应的总类
#t->type


#打印数据集
#print(i_v)
#print(i_t)

v_train,v_test,t_train,t_test=train_test_split(i_v,i_t,test_size=0.2,random_state=4)
#把整个集合分成学习的data和测试的两部分
#train:test=8:2
#这步split操作顺带将数据打乱打乱



#print(t_train)
#print(v_test)



knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(v_train,t_train)
# knn 是一个 KNeighborsClassifier 类的实例对象
# 它属于机器学习模型对象类型，具体来说是一个分类器（Classifier）对象
# 在训练完成后，它是一个已经训练好的K近邻分类模型，可以用于对新数据进行预测



Prediction=knn.predict(v_test)
#比对
print(Prediction)
print(t_test)
print("其中")
for i in range(len(t_test)):
    if knn.predict(v_test)[i]!=t_test[i]:
        print(f"第{i}朵花预测错误,该花特征值为:{v_test[i]},预测结果为:{knn.predict(v_test)[i]},正确结果为:{t_test[i]}")




print(knn.score(v_test,t_test))

