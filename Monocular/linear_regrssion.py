import numpy as np
from sklearn.linear_model import LinearRegression


x = [247, 180, 190, 150, 215, 230, 180, 151, 135, 180, 182, 144, 162, 160, 154, 135, 143, 143, 158, 133]
y = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

x = np.array(x).reshape((-1, 1))
y = np.array(y)

model = LinearRegression()

model.fit(x, y)
print(f"intercept: {round(model.intercept_, 1)}")
print(f"slope: {model.coef_}")
print(str(158*model.coef_ + model.intercept_))