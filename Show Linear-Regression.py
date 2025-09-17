from sklearn import datasets
#这种语法用于从模块中导入特定的函数、类或变量。

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
#导入整个个模块

x,y=datasets.make_regression(n_samples=100, n_features=1, noise=10)
plt.scatter(x,y)
plt.show()
