from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

loaded_data=fetch_california_housing()

data_x=loaded_data.data
data_y=loaded_data.target
#线性回归，故用x,y

model=LinearRegression()
#括号里可以增添参数

model.fit(data_x,data_y)
#开始训练

#print((data_x))
#数据比较多
print(model.predict(data_x[:4,:]))
#表示取data_x的前4行和所有列
print(data_y[:4])
#表示取data_y的前4个元素