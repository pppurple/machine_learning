import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

# ランダム値を生成
np.random
regdata = datasets.make_regression(100, 1, noise=20.0)

# 学習
lin = linear_model.LinearRegression()
lin.fit(regdata[0], regdata[1])
# 係数
print('Coefficients: \n', lin.coef_)
# 切片
print('Intercept: \n', lin.intercept_)
# スコア
print("score :", lin.score(regdata[0], regdata[1]))

# グラフ表示
xr = [-2.5, 2.5]
plt.plot(xr, lin.coef_ * xr + lin.intercept_)
plt.scatter(regdata[0], regdata[1])

plt.show()