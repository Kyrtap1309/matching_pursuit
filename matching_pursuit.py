import numpy as np
import pandas as pd

from sklearn.linear_model import OrthogonalMatchingPursuit

# Funkcja do obliczania korelacji między residualem a funkcją bazową z regularyzacją L2
def correlation_with_regularization(residuals, g, lambda_reg):
    return sum(residual * g(x) for x, residual in residuals) / (1 + lambda_reg)

def matching_pursuit(data_points, dictionary, lambda_reg=0.1, max_iter=5000, threshold=1e-5):
    residuals = [(x, y) for x, y in data_points]  # Początkowo residual = wartość funkcji
    approximation = {g.__name__: 0.0 for g in dictionary}  # Wstępna aproksymacja

    for _ in range(max_iter):
        correlations = [correlation_with_regularization(residuals, g, lambda_reg) for g in dictionary]
        best_index = np.argmax(np.abs(correlations))
        best_g = dictionary[best_index]
        alpha = correlations[best_index]

        # Sumujemy współczynniki dla wybranej funkcji
        approximation[best_g.__name__] += alpha

        # Aktualizujemy residual
        residuals = [(x, residual - alpha * best_g(x)) for x, residual in residuals]

        # Przerwanie, jeśli osiągnięto wymaganą dokładność
        residual_norm = np.sqrt(sum(residual**2 for _, residual in residuals))
        if residual_norm < threshold:
            break

    return approximation