"""descriptor_accuracy, per experiments/METRICS.md's 2026-09-13 definition.

z(phi) = (phi - mu) / sigma        # mu, sigma = TRAIN phi_t1 per-dim mean/std
descriptor_accuracy = 1 - rms(z(phi_pred) - z(phi_true)) / rms(z(phi_persist) - z(phi_true))
phi_persist = phi_t0 (persistence / do-nothing).
Pooled over all retained dims and all samples (rms over the flattened residual).
"""
from __future__ import annotations

import numpy as np


def fit_stats(phi_t1_train: np.ndarray, eps: float = 1e-8):
    mu = phi_t1_train.mean(axis=0)
    sigma = phi_t1_train.std(axis=0)
    sigma = np.maximum(sigma, eps)
    return mu, sigma


def zscore(phi: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (phi - mu) / sigma


def descriptor_accuracy(phi_pred: np.ndarray, phi_true: np.ndarray, phi_persist: np.ndarray,
                         mu: np.ndarray, sigma: np.ndarray) -> float:
    zp = zscore(phi_pred, mu, sigma)
    zt = zscore(phi_true, mu, sigma)
    zpe = zscore(phi_persist, mu, sigma)
    rms_model = np.sqrt(np.mean((zp - zt) ** 2))
    rms_persist = np.sqrt(np.mean((zpe - zt) ** 2))
    return float(1.0 - rms_model / rms_persist)
