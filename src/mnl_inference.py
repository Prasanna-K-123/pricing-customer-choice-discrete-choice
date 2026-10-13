"""Optional, analytic MNL inference with respondent-clustered covariance.

Historical fitting code and inference tables are not overwritten or imported.
All covariance statements are conditional on the chosen utility specification.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm


PARAMETERS = ("ASC_TRAIN", "B_TIME", "B_COST", "ASC_CAR")


@dataclass(frozen=True)
class ChoiceData:
    x: np.ndarray
    availability: np.ndarray
    choice: np.ndarray
    group: np.ndarray


def validate(x, availability, choice, group):
    x = np.asarray(x, dtype=float)
    availability = np.asarray(availability)
    choice = np.asarray(choice)
    group = np.asarray(group)
    if x.ndim != 3 or not all(x.shape) or x.shape[1] < 2:
        raise ValueError("x must be nonempty rows x alternatives x parameters")
    n, alternatives, _ = x.shape
    if availability.shape != (n, alternatives):
        raise ValueError("availability shape must match rows and alternatives")
    if choice.shape != (n,) or group.shape != (n,):
        raise ValueError("choice and group must contain exactly one value per row")
    if not np.isfinite(x).all():
        raise ValueError("all utility attributes must be finite")
    try:
        available_valid = np.isfinite(availability).all() and np.isin(availability, [0, 1]).all()
        choice_valid = np.isfinite(choice).all() and np.equal(choice, np.floor(choice)).all()
    except TypeError as exc:
        raise ValueError("availability and choice must be numeric") from exc
    if not available_valid:
        raise ValueError("availability must be boolean or 0/1")
    if choice.dtype.kind == "b" or not choice_valid or np.any(choice < 0) or np.any(choice >= alternatives):
        raise ValueError("choice must be a zero-based integer alternative")
    av = availability.astype(bool)
    y = choice.astype(np.int64)
    if not av.any(axis=1).all() or not av[np.arange(n), y].all():
        raise ValueError("each row must have an available chosen alternative")
    if group.dtype.kind in "fc" and not np.isfinite(group).all():
        raise ValueError("respondent identifiers must not be missing or nonfinite")
    if any(v is None or (isinstance(v, str) and not v) for v in group):
        raise ValueError("respondent identifiers must not be missing")
    try:
        np.unique(group)
    except (TypeError, ValueError) as exc:
        raise ValueError("respondent identifiers must have a comparable common type") from exc
    return ChoiceData(x.copy(), av.copy(), y.copy(), group.copy())


def from_swissmetro(frame):
    needed = ["ID", "CHOICE", "GA", "SP", "TRAIN_AV", "SM_AV", "CAR_AV",
              "TRAIN_TT", "SM_TT", "CAR_TT", "TRAIN_CO", "SM_CO", "CAR_CO"]
    if any(name not in frame.columns for name in needed) or frame[needed].isna().any().any():
        raise ValueError("required Swissmetro columns must be present and nonmissing")
    x = np.zeros((len(frame), 3, 4), dtype=float)
    x[:, 0, 0] = 1
    x[:, 2, 3] = 1
    x[:, :, 1] = frame[["TRAIN_TT", "SM_TT", "CAR_TT"]].to_numpy(float) / 100
    x[:, :, 2] = frame[["TRAIN_CO", "SM_CO", "CAR_CO"]].to_numpy(float) / 100
    x[:, :2, 2] *= frame["GA"].eq(0).to_numpy()[:, None]
    av = frame[["TRAIN_AV", "SM_AV", "CAR_AV"]].to_numpy()
    av[:, [0, 2]] *= frame["SP"].ne(0).to_numpy()[:, None]
    return validate(x, av, frame["CHOICE"].to_numpy() - 1, frame["ID"].to_numpy())


def likelihood_parts(beta, data):
    beta = np.asarray(beta, dtype=float)
    if beta.shape != (data.x.shape[2],) or not np.isfinite(beta).all():
        raise ValueError("beta must contain a finite value for every parameter")
    with np.errstate(over="ignore", invalid="ignore"):
        utility = np.einsum("nak,k->na", data.x, beta)
    if not np.isfinite(utility).all():
        raise ValueError("utility overflow or nonfinite values")
    masked = np.where(data.availability, utility, -np.inf)
    log_probability = masked - logsumexp(masked, axis=1, keepdims=True)
    probability = np.exp(log_probability)
    expected_x = np.einsum("na,nak->nk", probability, data.x)
    scores = data.x[np.arange(len(data.choice)), data.choice] - expected_x
    centered = data.x - expected_x[:, None, :]
    information = np.einsum("na,nak,nal->kl", probability, centered, centered)
    log_likelihood = log_probability[np.arange(len(data.choice)), data.choice].sum()
    return float(log_likelihood), scores, information, probability


def fit(data):
    k = data.x.shape[2]
    objective = lambda b: -likelihood_parts(b, data)[0]
    gradient = lambda b: -likelihood_parts(b, data)[1].sum(axis=0)
    hessian = lambda b: likelihood_parts(b, data)[2]
    result = minimize(objective, np.zeros(k), jac=gradient, hess=hessian,
                      method="trust-exact", options={"gtol": 1e-8, "maxiter": 1000})
    gradient_norm = float(np.max(np.abs(gradient(result.x))))
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all() or gradient_norm > 1e-6:
        raise ValueError(f"fit does not meet the registered stationarity criterion: {result.message}; {gradient_norm}")
    return result.x.copy(), {"optimizer_success": bool(result.success),
                            "optimizer_message": str(result.message),
                            "gradient_infinity_norm": gradient_norm,
                            "iterations": int(result.nit)}


def covariance(beta, data):
    ll, scores, information, _ = likelihood_parts(beta, data)
    eigenvalues = np.linalg.eigvalsh(information)
    if not np.isfinite(eigenvalues).all() or eigenvalues[-1] <= 0 or eigenvalues[0] <= 1e-12 * eigenvalues[-1]:
        raise ValueError("information matrix is singular or numerically unidentified")
    groups, inverse = np.unique(data.group, return_inverse=True)
    n, k = scores.shape
    g = len(groups)
    if g < 2 or n <= k:
        raise ValueError("cluster inference requires at least two groups and N>K")
    inverse_information = np.linalg.inv(information)
    grouped_scores = np.zeros((g, k))
    np.add.at(grouped_scores, inverse, scores)
    row_sandwich = inverse_information @ (scores.T @ scores) @ inverse_information
    cluster_uncorrected = inverse_information @ (grouped_scores.T @ grouped_scores) @ inverse_information
    factor = g / (g - 1) * (n - 1) / (n - k)
    return {"log_likelihood": ll, "information": information,
            "information_eigenvalues": eigenvalues,
            "conventional": inverse_information, "row_sandwich": row_sandwich,
            "cluster_uncorrected": cluster_uncorrected,
            "cluster_corrected": factor * cluster_uncorrected,
            "finite_sample_factor": factor, "rows": n, "clusters": g}


def inference_rows(beta, estimates, names=PARAMETERS):
    if len(names) != len(beta):
        raise ValueError("one parameter name is required per coefficient")
    rows = []
    for index, name in enumerate(names):
        row = {"parameter": name, "estimate": float(beta[index])}
        for method in ["conventional", "row_sandwich", "cluster_uncorrected", "cluster_corrected"]:
            value = float(estimates[method][index, index])
            if not np.isfinite(value) or value <= 0:
                raise ValueError("inference requires strictly positive finite marginal variance")
            se = float(np.sqrt(value))
            z = float(beta[index] / se)
            row[method + "_se"] = se
            row[method + "_z"] = z
            row[method + "_normal_p"] = float(2 * norm.sf(abs(z)))
        rows.append(row)
    return rows
