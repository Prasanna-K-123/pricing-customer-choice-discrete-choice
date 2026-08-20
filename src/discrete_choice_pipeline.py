"""Swissmetro discrete-choice modeling and pricing-sensitivity pipeline."""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

DATA_URL = "https://transp-or.epfl.ch/data/swissmetro.dat"
OFFICIAL_BIOGEME_LL = -5331.252


def load_data():
    r = requests.get(DATA_URL, timeout=60)
    r.raise_for_status()
    raw = pd.read_csv(io.BytesIO(r.content), sep="\t")
    df = raw.loc[raw["PURPOSE"].isin([1, 3]) & raw["CHOICE"].ne(0)].copy().reset_index(drop=True)
    df["TRAIN_COST"] = np.where(df["GA"].eq(0), df["TRAIN_CO"], 0.0)
    df["SM_COST"] = np.where(df["GA"].eq(0), df["SM_CO"], 0.0)
    df["TRAIN_AV_SP"] = df["TRAIN_AV"] * df["SP"].ne(0).astype(int)
    df["CAR_AV_SP"] = df["CAR_AV"] * df["SP"].ne(0).astype(int)
    for c in ["TRAIN_TT", "SM_TT", "CAR_TT"]:
        df[c + "_S"] = df[c] / 100.0
    df["TRAIN_COST_S"] = df["TRAIN_COST"] / 100.0
    df["SM_COST_S"] = df["SM_COST"] / 100.0
    df["CAR_COST_S"] = df["CAR_CO"] / 100.0
    assert len(raw) == 10728 and len(df) == 6768
    return raw, df


def availability_matrix(frame):
    return np.column_stack([
        frame["TRAIN_AV_SP"].to_numpy(float),
        frame["SM_AV"].to_numpy(float),
        frame["CAR_AV_SP"].to_numpy(float),
    ])


def utility_matrix(params, frame, sm_cost_multiplier=1.0):
    asc_train, b_time, b_cost, asc_car = params
    train = asc_train + b_time * frame["TRAIN_TT_S"].to_numpy() + b_cost * frame["TRAIN_COST_S"].to_numpy()
    sm = b_time * frame["SM_TT_S"].to_numpy() + b_cost * frame["SM_COST_S"].to_numpy() * sm_cost_multiplier
    car = asc_car + b_time * frame["CAR_TT_S"].to_numpy() + b_cost * frame["CAR_COST_S"].to_numpy()
    return np.column_stack([train, sm, car])


def choice_probabilities(params, frame, sm_cost_multiplier=1.0):
    u = utility_matrix(params, frame, sm_cost_multiplier)
    av = availability_matrix(frame)
    u = np.where(av > 0, u, -1e12)
    u = u - u.max(axis=1, keepdims=True)
    eu = np.exp(u) * av
    return eu / eu.sum(axis=1, keepdims=True)


def negative_log_likelihood(params, frame):
    p = choice_probabilities(params, frame)
    chosen = frame["CHOICE"].to_numpy(int) - 1
    return -np.log(np.clip(p[np.arange(len(frame)), chosen], 1e-15, 1.0)).sum()


def fit_mnl(frame):
    return minimize(
        negative_log_likelihood,
        x0=np.zeros(4),
        args=(frame,),
        method="BFGS",
        options={"gtol": 1e-7, "maxiter": 1000},
    )


def respondent_holdout(df):
    ids = df["ID"].drop_duplicates().to_numpy()
    train_ids, test_ids = train_test_split(ids, test_size=0.20, random_state=2026)
    train = df[df["ID"].isin(train_ids)].copy()
    test = df[df["ID"].isin(test_ids)].copy()
    assert set(train["ID"]).isdisjoint(set(test["ID"]))
    result = fit_mnl(train)
    probs = choice_probabilities(result.x, test)
    y = test["CHOICE"].to_numpy(int) - 1
    mnl_ll = log_loss(y, probs, labels=[0, 1, 2])
    mnl_acc = accuracy_score(y, np.argmax(probs, axis=1))

    shares = train["CHOICE"].value_counts(normalize=True).reindex([1, 2, 3], fill_value=0).to_numpy()
    base = np.tile(shares, (len(test), 1)) * availability_matrix(test)
    base /= base.sum(axis=1, keepdims=True)
    base_ll = log_loss(y, base, labels=[0, 1, 2])
    base_acc = accuracy_score(y, np.argmax(base, axis=1))
    return train, test, base_ll, mnl_ll, base_acc, mnl_acc


def pricing_scenarios(df, params):
    pricing_df = df[df["GA"].eq(0)].copy()
    rows = []
    for m in np.round(np.arange(0.60, 1.401, 0.05), 2):
        p_sm = choice_probabilities(params, pricing_df, sm_cost_multiplier=m)[:, 1]
        fare = pricing_df["SM_COST"].to_numpy() * m
        rows.append({
            "fare_multiplier": m,
            "predicted_sm_share": p_sm.mean(),
            "expected_fare_revenue_proxy": np.mean(p_sm * fare),
        })
    scenarios = pd.DataFrame(rows)
    base = scenarios.loc[scenarios["fare_multiplier"].eq(1.0)].iloc[0]
    scenarios["share_index_vs_base"] = scenarios["predicted_sm_share"] / base["predicted_sm_share"] * 100
    scenarios["fare_revenue_index_vs_base"] = scenarios["expected_fare_revenue_proxy"] / base["expected_fare_revenue_proxy"] * 100
    s90 = scenarios.loc[scenarios["fare_multiplier"].eq(0.90), "predicted_sm_share"].iloc[0]
    s110 = scenarios.loc[scenarios["fare_multiplier"].eq(1.10), "predicted_sm_share"].iloc[0]
    elasticity = np.log(s110 / s90) / np.log(1.10 / 0.90)
    best = scenarios.sort_values("expected_fare_revenue_proxy", ascending=False).iloc[0]
    return scenarios, base, elasticity, best


def main():
    raw, df = load_data()
    result = fit_mnl(df)
    ll = -result.fun
    params = result.x
    assert abs(ll - OFFICIAL_BIOGEME_LL) < 0.05

    asc_train, b_time, b_cost, asc_car = params
    vot_hour = (b_time / b_cost) * 60
    wtp_10min = (b_time / b_cost) * 10

    train, test, base_ll, mnl_ll, base_acc, mnl_acc = respondent_holdout(df)
    scenarios, base, elasticity, best = pricing_scenarios(df, params)

    out = Path("results")
    out.mkdir(exist_ok=True)
    pd.DataFrame({
        "parameter": ["ASC_TRAIN", "B_TIME", "B_COST", "ASC_CAR"],
        "estimate": params,
    }).to_csv(out / "mnl_parameter_estimates.csv", index=False)
    pd.DataFrame({
        "model": ["Training-share baseline", "Multinomial logit"],
        "test_log_loss": [base_ll, mnl_ll],
        "test_accuracy": [base_acc, mnl_acc],
    }).to_csv(out / "respondent_holdout_validation.csv", index=False)
    scenarios.to_csv(out / "swissmetro_pricing_scenarios.csv", index=False)

    metrics = pd.DataFrame({"metric": [
        "usable_choice_observations", "unique_respondents", "full_sample_log_likelihood",
        "official_biogeme_log_likelihood", "absolute_ll_validation_difference",
        "value_of_time_chf_per_hour", "willingness_to_pay_10min_chf",
        "holdout_baseline_log_loss", "holdout_mnl_log_loss", "holdout_log_loss_improvement",
        "holdout_baseline_accuracy", "holdout_mnl_accuracy", "base_sm_share_non_ga",
        "sm_arc_fare_elasticity", "best_tested_fare_multiplier", "best_tested_fare_revenue_index",
    ], "value": [
        len(df), df["ID"].nunique(), ll, OFFICIAL_BIOGEME_LL, abs(ll - OFFICIAL_BIOGEME_LL),
        vot_hour, wtp_10min, base_ll, mnl_ll, 1 - mnl_ll / base_ll, base_acc, mnl_acc,
        base["predicted_sm_share"], elasticity, best["fare_multiplier"], best["fare_revenue_index_vs_base"],
    ]})
    metrics.to_csv(out / "project7_metrics.csv", index=False)

    print(f"Usable observations: {len(df):,} | Respondents: {df['ID'].nunique():,}")
    print(f"Log likelihood: {ll:.6f} | benchmark difference: {abs(ll - OFFICIAL_BIOGEME_LL):.6f}")
    print(f"Value of time: {vot_hour:.2f} CHF/hour | WTP 10 min: {wtp_10min:.2f} CHF")
    print(f"Holdout accuracy: {mnl_acc:.2%} vs {base_acc:.2%} baseline")
    print(f"Holdout log-loss improvement: {1 - mnl_ll / base_ll:.2%}")
    print(f"Own-fare elasticity: {elasticity:.3f}")
    print(f"Best tested fare-revenue proxy: {best['fare_multiplier']:.2f}x, index {best['fare_revenue_index_vs_base']:.2f}")


if __name__ == "__main__":
    main()
