import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import numpy as np


# QUESTION 7

data = pd.read_csv('measurements.csv')

X = data.iloc[:, 0].values.reshape(-1, 1) #[1, 2, 3, 4] -> [[1], [2], [3], [4]]
y = data.iloc[:, 1].values

model = LinearRegression()
model.fit(X, y)


y_pred = model.predict(X)

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Data points', s=50, alpha=0.7)
plt.plot(X, y_pred, color='red', linewidth=2, label='Linear regression line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Linear Regression (R² = {r2:.4f}, MSE = {mse:.4f}, coef = {model.coef_[0]:.4f}, intercept = {model.intercept_:.4f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('regression_plot.png', dpi=100, bbox_inches='tight')
plt.show()


# QUESTION 8

degree_grid = np.arange(1, 26)

plt.figure(figsize=(12, 7))
plt.scatter(X, y, color='black', label='Data points')

x = np.linspace(0, 15, num=200)
empirical_risks = []


for degree in degree_grid:
	poly_features = PolynomialFeatures(degree=degree, include_bias=False)
	X_poly = poly_features.fit_transform(X)

	poly_reg_model = LinearRegression()
	poly_reg_model.fit(X_poly, y)

	y_pred = poly_reg_model.predict(X_poly)
	mse = mean_squared_error(y, y_pred)
	empirical_risks.append(mse)
	r2 = r2_score(y, y_pred)

	p = np.append(np.flip(poly_reg_model.coef_), poly_reg_model.intercept_)
	y3 = np.polyval(p, x)

	plt.plot(x, y3, linewidth=2, label=f'Degree {degree} (R2={r2:.3f}, MSE={mse:.3f})')

best_idx = int(np.argmin(empirical_risks))
best_degree = int(degree_grid[best_idx])
best_mse = float(empirical_risks[best_idx])

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Polynomial Regression with Degrees 1-25')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('polynomial_regression_plot.png', dpi=100, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(degree_grid, empirical_risks, marker='o', linewidth=2, color='tab:green', label='Empirical risk (train MSE)')
plt.scatter([best_degree], [best_mse], color='red', s=80, zorder=3, label=f'Min at degree {best_degree}')
plt.xlabel('Polynomial degree')
plt.ylabel('Empirical risk (MSE)')
plt.title('Empirical Risk Minimization Curve')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('erm_curve.png', dpi=100, bbox_inches='tight')
plt.show()





