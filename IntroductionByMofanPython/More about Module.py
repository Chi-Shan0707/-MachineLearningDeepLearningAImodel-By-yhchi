from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

loaded_data=fetch_california_housing()

data_x=loaded_data.data
data_y=loaded_data.target
#线性回归，故用x,y

#model=LinearRegression()
#括号里可以增添参数

model=LinearRegression(n_jobs=2,fit_intercept=False)

model.fit(data_x,data_y)
#开始训练

#print((data_x))
#数据比较多
#print(model.predict(data_x[:4,:]))
#表示取data_x的前4行和所有列
#print(data_y[:4])
#表示取data_y的前4个元素

print(model.coef_)
#coefficient
print(model.intercept_)
#intercept

#parameter
print(model.get_params())
"""
n_job:使用的cpu核数

fit_intercept 此模型的截距。如果设置为False，则不会计算截距（即回归线通过原点）。

positive含义：是否强制所有回归系数为正数;默认值：False;
         作用：
        当设置为True时，强制回归系数(coef_)为非负数
         这在某些特定场景下很有用，比如在物理或经济学模型中，某些变量的影响必须是正向的
normalize: 布尔值，指定是否在回归前对特征进行标准化（注意：在新版本sklearn中已被弃用）。

copy_X :
含义：指定是否在计算前复制输入特征矩阵X
默认值：True
作用：
当设置为True时，X会被复制，原始数据不会被修改
当设置为False时，允许在计算过程中覆盖原始X数据，这可以节省内存但会修改原始数据
使用场景：当内存资源紧张且可以接受修改原始数据时，可以设置为False

tol：
tol（tolerance，容差）参数出现在使用迭代优化算法的模型中，用于控制算法何时停止迭代。具体来说：

收敛条件：当算法的损失函数变化小于 tol 值时，算法停止迭代
默认值：许多算法的默认 tol 值是 1e-06（即 0.000001）
精度控制：较小的 tol 值意味着算法会运行更多迭代以获得更精确的结果
"""
print(model.score(data_x,data_y))

"""

对model学到的东西进行打分
R²（决定系数）分数
##
R²的计算公式是：

R² = 1 - (SS_res / SS_tot)

其中：

SS_res（残差平方和）= Σ(y - y')²，即实际值与预测值之差的平方和
SS_tot（总平方和）= Σ(y - ȳ)²，即实际值与均值之差的平方和
"""

