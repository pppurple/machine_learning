import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
from pprint import pprint

diabetes = datasets.load_diabetes()

# BMIのデータを取得
diabetes_bmi = diabetes.data[:, np.newaxis, 2]

# トレーニング用データとテスト用データに分割
data_train = diabetes_bmi[:-20]
target_train = diabetes.target[:-20]
data_test = diabetes_bmi[-20:]
target_test = diabetes.target[-20:]

# 線形回帰オブジェクト生成
lin = linear_model.LinearRegression()

# 学習
lin.fit(data_train, target_train)

# 係数
print('Coefficients: \n', lin.coef_)
# 切片
print('Intercept: \n', lin.intercept_)
# 平均二乗誤差(標準偏差)
print("Residual sum of squares: %.2f" % np.mean((lin.predict(data_test) - target_test) ** 2))
# スコア
print("Variance score :", lin.score(data_test, target_test))

# 散布図
plt.scatter(data_test, target_test, color='black')
# 線図
plt.plot(data_test, lin.predict(data_test), color='blue')

plt.show()
