import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Генерация искусственных данных
np.random.seed(42)
X = np.linspace(10, 100, 20)  # Цены товара
noise = np.random.normal(0, 8, size=X.shape)
Y = 150 - 1.2 * X + noise     # Спрос с шумом

# Линейная регрессия
slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
Y_pred = intercept + slope * X

# Результаты
print(f"Наклон (slope): {slope:.3f}")
print(f"Свободный член (intercept): {intercept:.3f}")
print(f"Коэффициент детерминации R²: {r_value**2:.3f}")

# Визуализация
plt.figure(figsize=(10, 5))
plt.scatter(X, Y, color='blue', label='Фактический спрос')
plt.plot(X, Y_pred, color='red', label='Линия регрессии')
plt.xlabel('Цена товара')
plt.ylabel('Спрос')
plt.title('Прогнозирование спроса на товар')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
