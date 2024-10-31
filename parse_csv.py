import pandas as pd

plik = "sweeds_full.csv"
df = pd.read_csv(plik)

# Definicja kolumn do wybrania w słowniku
kolumny_dict = {
    "cn": ["ALPHA", "BETA", "DEFL_1_1", "DEFL_1_2", "DEFL_1_3", "DEFL_1_4", "CN"],  # Siła w Z
    "cm": ["ALPHA", "BETA", "DEFL_1_1", "DEFL_1_2", "DEFL_1_3", "DEFL_1_4", "CM"],  # Moment wokół Y
    "ca": ["ALPHA", "BETA", "DEFL_1_1", "DEFL_1_2", "DEFL_1_3", "DEFL_1_4", "CA"],  # Siła w X
    "cy": ["ALPHA", "BETA", "DEFL_1_1", "DEFL_1_2", "DEFL_1_3", "DEFL_1_4", "CY"],  # Siła w Y
    "cln": ["ALPHA", "BETA", "DEFL_1_1", "DEFL_1_2", "DEFL_1_3", "DEFL_1_4", "CLN"],  # Moment wokol Z
    "cll": ["ALPHA", "BETA", "DEFL_1_1", "DEFL_1_2", "DEFL_1_3", "DEFL_1_4", "CLL"],  # Moment wokol X
}

# Iteracja po słowniku, wybór kolumn i zapis do plików CSV
for nazwa, kolumny in kolumny_dict.items():
    df_wybrane = df[kolumny]
    df_wybrane.to_csv(f"{nazwa}.csv", index=False)

print("Nowe pliki CSV zostały zapisane.")
