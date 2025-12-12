from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def compute_importance_precise(X, y, n_repeats=5, perm_repeats=20):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    model = RandomForestRegressor(
        n_estimators=800,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    )

    cv = RepeatedKFold(n_splits=5, n_repeats=n_repeats, random_state=42)
    r2_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='r2')
    print(f"CV R²: {r2_scores.mean():.3f} ± {r2_scores.std():.3f}")

    model.fit(X_scaled, y)


    perm = permutation_importance(
        model,
        X_scaled,
        y,
        n_repeats=perm_repeats,
        random_state=42,
        n_jobs=-1,
    )

    df = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    })


    df["importance_mean_clipped"] = np.clip(df["importance_mean"], a_min=0, a_max=None)
    total = df["importance_mean_clipped"].sum()
    if total > 0:
        df["percentage"] = (df["importance_mean_clipped"] / total * 100).round(3)
    else:
        df["percentage"] = 0.0

    df = df.sort_values("percentage", ascending=False).reset_index(drop=True)
    return df.round(3)

def compute_importance(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=500, random_state=42)
    model.fit(X_scaled, y)

    importance = model.feature_importances_
    df = pd.DataFrame({
        "feature": X.columns,
        "importance": importance
    })
    df["percentage"] = df["importance"] / df["importance"].sum() * 100
    df = df.sort_values("percentage", ascending=False)

    return df
