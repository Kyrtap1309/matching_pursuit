import numpy as np
import pandas as pd

filename = "cy.csv"  # Wpisz odpowiedni csv (jeden csv --> jedna funkcja aerodynamiczna)
data = pd.read_csv(filename).iloc[:20]  # Wybieramy pierwsze 20 wierszy (dla przykładu)

# Tworzenie punktów danych bez skalowania
data_points = [
    (row[:6], row[6])  # Pierwsze 6 kolumn jako wektor i ostatnia kolumna jako wartość funkcji
    for row in data.values
]

# Stopnie na radiany
def to_radians(x):
    return np.radians(x)

# Rozszerzony słownik funkcji trygonometrycznych i ich kombinacji z normalizacją
# Funkcje bazowe
def g1(x): return np.sin(x[0]) / 10
def g2(x): return np.cos(x[0]) / 10
def g3(x): return np.sin(x[1]) / 10
def g4(x): return np.cos(x[1]) / 10
def g5(x): return np.sin(x[2]) / 10
def g6(x): return np.cos(x[2]) / 10
def g7(x): return np.sin(x[3]) / 10
def g8(x): return np.cos(x[3]) / 10
def g9(x): return np.sin(x[4]) / np.sqrt(len(data_points))
def g10(x): return np.cos(x[4]) / np.sqrt(len(data_points))
def g11(x): return np.sin(x[5]) / np.sqrt(len(data_points))
def g12(x): return np.cos(x[5]) / np.sqrt(len(data_points))

# Kombinacje funkcji
def g13(x): return (np.sin(x[0]) * np.cos(x[1])) / np.sqrt(len(data_points))
def g14(x): return (np.sin(x[2]) * np.cos(x[3])) / np.sqrt(len(data_points))
def g15(x): return (np.sin(x[4]) * np.cos(x[5])) / np.sqrt(len(data_points))

dictionary = [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12,
              g13, g14, g15]
function_names = {f.__name__: f for f in dictionary}

# Funkcja do obliczania korelacji między residualem a funkcją bazową z regularyzacją L2
def correlation_with_regularization(residuals, g, lambda_reg):
    return sum(residual * g(x) for x, residual in residuals) / (1 + lambda_reg)

# Algorytm matching pursuit z regularyzacja L2
def matching_pursuit(data_points, dictionary, lambda_reg=0.1, max_iter=20, threshold=1e-5):
    residuals = [(x, y) for x, y in data_points]  # Początkowo residual = wartość funkcji
    approximation = {g.__name__: 0.0 for g in dictionary}  # Wstepna aproksymacja

    for _ in range(max_iter):
        correlations = [correlation_with_regularization(residuals, g, lambda_reg) for g in dictionary]
        best_index = np.argmax(np.abs(correlations))
        best_g = dictionary[best_index]
        alpha = correlations[best_index]

        # Sumujemy współczynniki dla wybranej funkcji
        approximation[best_g.__name__] += alpha

        # Aktualizujemy residual
        residuals = [(x, residual - alpha * best_g(x)) for x, residual in residuals]

        # Zbreakuj jak osiagnieto zbieznosc
        residual_norm = np.sqrt(sum(residual**2 for _, residual in residuals))
        if residual_norm < threshold:
            break

    return approximation

# Obliczenie wartości aproksymacji oraz błędu względnego
def evaluate_and_compare(data_points, approximation):
    results = []
    relative_errors = []
    for x, true_value in data_points:
        approx_value = sum(alpha * function_names[func_name](x) for func_name, alpha in approximation.items())
        relative_error = np.abs((true_value - approx_value) / true_value) * 100 if true_value != 0 else None
        results.append((approx_value, true_value, relative_error))
        if relative_error is not None:  # Pomijamy przypadki, gdzie błąd względny jest niemożliwy do obliczenia
            relative_errors.append(relative_error)
    
    # Obliczenie średniego błędu względnego
    average_relative_error = np.mean(relative_errors) if relative_errors else None
    return results, average_relative_error

# Przykład użycia
approximation = matching_pursuit(data_points, dictionary, lambda_reg=0.1)
print("Aproksymacja jako kombinacja funkcji ze słownika:")
for func_name, alpha in approximation.items():
    if abs(alpha) > 1e-5:  # Ignorowanie zera
        print(f"{alpha:.4f} * {func_name}")

# Porównanie z wartościami rzeczywistymi, wyliczenie błędów względnych oraz średniego błędu względnego
results, average_relative_error = evaluate_and_compare(data_points, approximation)

# Wyświetlanie wyników
print("\nWyniki aproksymacji i błędów względnych:")
for approx_value, true_value, relative_error in results:
    print(f"Aproksymacja: {approx_value:.4f}, Wartość rzeczywista: {true_value:.4f}, Błąd względny: {relative_error:.2f}%")

# Wyświetlanie średniego błędu względnego
if average_relative_error is not None:
    print(f"\nŚredni błąd względny: {average_relative_error:.2f}%")
else:
    print("\nŚredni błąd względny: Brak wartości rzeczywistych do obliczenia błędu.")