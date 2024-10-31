import numpy as np
import pandas as pd

from matching_pursuit import matching_pursuit

N = 20 #Liczba wczytywanych wierszy
file_name = "cy"

# Ponownie definiujemy przekształcanie stopni na radiany
def to_radians(x):
    return np.radians(x)

# Funkcje bazowe i ich kombinacje
# Rozszerzony słownik funkcji trygonometrycznych i ich kombinacji z normalizacją
def g1(x): return np.sin(to_radians(x[0])) / np.sqrt(N)
def g2(x): return np.cos(to_radians(x[0])) / np.sqrt(N)
def g3(x): return np.sin(to_radians(x[1])) / np.sqrt(N)
def g4(x): return np.cos(to_radians(x[1])) / np.sqrt(N)
def g5(x): return np.sin(to_radians(x[2])) / np.sqrt(N)
def g6(x): return np.cos(to_radians(x[2])) / np.sqrt(N)
def g7(x): return np.sin(to_radians(x[3])) / np.sqrt(N)
def g8(x): return np.cos(to_radians(x[3])) / np.sqrt(N)
def g9(x): return np.sin(to_radians(x[4])) / np.sqrt(N)
def g10(x): return np.cos(to_radians(x[4])) / np.sqrt(N)
def g11(x): return np.sin(to_radians(x[5])) / np.sqrt(N)
def g12(x): return np.cos(to_radians(x[5])) / np.sqrt(N)

# Kombinacje funkcji
def g13(x): return (np.sin(to_radians(x[0])) * np.cos(to_radians(x[1]))) / np.sqrt(N)
def g14(x): return (np.sin(to_radians(x[2])) * np.cos(to_radians(x[3]))) / np.sqrt(N)
def g15(x): return (np.sin(to_radians(x[4])) * np.cos(to_radians(x[5]))) / np.sqrt(N)
def g16(x): return (np.sin(to_radians(x[0])) * np.sin(to_radians(x[2]))) / np.sqrt(N)
def g17(x): return (np.cos(to_radians(x[1])) * np.cos(to_radians(x[3]))) / np.sqrt(N)
def g18(x): return (np.sin(to_radians(x[0]))**2) / np.sqrt(N)
def g19(x): return (np.cos(to_radians(x[1]))**2) / np.sqrt(N)
def g20(x): return (np.sin(to_radians(x[2]))**2) / np.sqrt(N)
def g21(x): return (np.cos(to_radians(x[3]))**2) / np.sqrt(N)
def g22(x): return (np.sin(to_radians(x[4]))**2) / np.sqrt(N)
def g23(x): return (np.cos(to_radians(x[5]))**2) / np.sqrt(N)
def g24(x): return (np.sin(to_radians(x[0] + x[1] + x[2]))) / np.sqrt(N)
def g25(x): return (np.cos(to_radians(x[3] + x[4] + x[5]))) / np.sqrt(N)
def g26(x): return (np.sin(to_radians(x[0]) * to_radians(x[1]))) / np.sqrt(N)
def g27(x): return (np.cos(to_radians(x[2]) * to_radians(x[3]))) / np.sqrt(N)


# Słownik funkcji bazowych
dictionary = [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12,
              g13, g14, g15, g16, g17, g18, g19, g20, g21, g22, g23, g24, g25, g26, g27]
function_names = {f.__name__: f for f in dictionary}

# Funkcja aproksymacji
def evaluate_approximation(x, approximation_coefficients):
    result = 0
    for func_name, alpha in approximation_coefficients.items():
        if func_name in function_names:
            result += alpha * function_names[func_name](x)
    return result

# Przykład funkcji testującej
def test_approximation():
    # Wczytanie danych z CSV
    data = pd.read_csv(file_name+".csv").iloc[:N]  # Wczytujemy 20 wierszy
    data_points = [
    (row[:6], row[6])  # Pierwsze 6 kolumn jako wektor i ostatnia kolumna jako wartość funkcji
    for row in data.values
]  # Pierwsze kolumny to wartości wejściowe
    target_values = data.iloc[:, -1].values  # Ostatnia kolumna to wartości docelowe

    # Wykonanie Matching Pursuit
    approximation_coefficients = matching_pursuit(data_points, dictionary)

    print(approximation_coefficients)

    # Obliczenie wartości aproksymacji dla każdego wiersza
    approximations = [
        evaluate_approximation(x[0], approximation_coefficients)
        for x in data_points
    ]

    relative_errors = [
        abs((target - approx) / target) * 100 if target != 0 else np.nan
        for target, approx in zip(target_values, approximations)
    ]

    # Dodanie kolumny z wynikami aproksymacji do DataFrame
    data[f'Approx_{file_name}'] = approximations
    data['Relative_Error_%'] = relative_errors

    # Zapis do pliku CSV z wartościami aproksymacji
    data.to_csv(file_name + "_approx.csv", index=False)

    # Test: Porównanie wyników
    print("Porównanie wartości rzeczywistych i aproksymacji (pierwsze 10 wierszy):")
    print(data[[f'Approx_{file_name}', data.columns[-3], 'Relative_Error_%']].head(10))
    print(f"Sredni blad wzgledny dla {N} wierszy: {np.mean([relative_error for relative_error in relative_errors if relative_error != np.nan])}")

# Uruchomienie testu
test_approximation()
