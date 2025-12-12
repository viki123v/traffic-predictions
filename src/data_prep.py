import pandas as pd

def load_and_clean(path):
    df = pd.read_csv(path)
    X = df.iloc[:, 6:-10]
    y = df.iloc[:, -10:].sum(axis=1)
    return X, y



STATIC_COLS = [
    "RB",
    "R_NUMBER",
    "S_NUMBER",
    "S_NAME",
    "Datum",
    "LENGTH SECTION",
    "Ave_Rain",
    "Max_Snow",
    "Ave_Temp",
    "Max_Temp",
    "Min_Temp",
    "Аve_Height",
    "LIMIT",
    "Kr.Kar.",
    "Ave_Rad",
    "Lat_Force",
    "Ave_Inc",
    "SKIDRES",
    "RDB_RUT ",
    "IRI",
    "PCI",
    "KON.NAK.",
    "SSD H",
    "SSD V",
    "V sign",
    "H sign",
    "K.Int. P.T",
    "K.Bridges",
]


def build_avg_dataset(df: pd.DataFrame,
                      years=range(2014, 2024)) -> pd.DataFrame:

    df = df.copy()


    pgds_cols = [f"PGDS {y}" for y in years if f"PGDS {y}" in df.columns]
    wi_cols   = [f"Wi {y}"   for y in years if f"Wi {y}"   in df.columns]


    if pgds_cols:
        df["PGDS_AVG"] = df[pgds_cols].mean(axis=1, skipna=True)
    else:
        df["PGDS_AVG"] = pd.NA

    if wi_cols:
        df["Wi_AVG"] = df[wi_cols].mean(axis=1, skipna=True)
    else:
        df["Wi_AVG"] = pd.NA

    cols_out = STATIC_COLS + ["PGDS_AVG", "Wi_AVG"]

    cols_out = [c for c in cols_out if c in df.columns]

    return df[cols_out].copy()


def build_yearly_dataset(df: pd.DataFrame,
                         year: int) -> pd.DataFrame:

    df = df.copy()

    pgds_col_src = f"PGDS {year}"
    wi_col_src   = f"Wi {year}"

    if pgds_col_src not in df.columns:
        raise ValueError(f"Column '{pgds_col_src}' not found in df.columns")
    if wi_col_src not in df.columns:
        raise ValueError(f"Column '{wi_col_src}' not found in df.columns")

    cols_out = STATIC_COLS + [pgds_col_src, wi_col_src]
    cols_out = [c for c in cols_out if c in df.columns]

    df_year = df[cols_out].copy()


    df_year = df_year.rename(columns={
        pgds_col_src: f"PGDS_{year}",
        wi_col_src:   f"Wi_{year}",
    })

    return df_year

import pandas as pd

def load_and_clean_year(csv_path: str, target_col: str):


    df = pd.read_csv(csv_path)


    df = df.dropna(subset=[target_col])

    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]

    feature_cols = [c for c in numeric_cols if c != target_col]

    X = df[feature_cols].copy()
    y = df[target_col].astype(float)

    return X, y
