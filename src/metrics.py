"""
src/metrics.py
Forensic Evaluation Metrics: Peak-to-Average Power Ratio (PAPR),
ROC-AUC, and threshold-based decision boundaries.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score


def compute_papr(radial_profile, r_min=10, eps=1e-8):
    """
    Calculates Peak-to-Average Power Ratio (PAPR) over the active high-frequency spectrum.

    Parameters:
        radial_profile (np.ndarray or list): 1D radial power profile S_R(r).
        r_min (int): Minimum radial index to suppress low-frequency DC remnants.
        eps (float): Numerical stability constant.

    Returns:
        float: Computed scalar PAPR value.
    """
    spectrum = np.asarray(radial_profile, dtype=np.float32)
    active = spectrum[r_min:]
    if len(active) == 0:
        return 0.0

    peak_power = np.max(active)
    avg_power = np.mean(active) + eps
    return float(peak_power / avg_power)


def evaluate_forensic_classifier(real_paprs, fake_paprs):
    """
    Evaluates detection performance given PAPR scores of pristine and fake images.

    Parameters:
        real_paprs (list or np.ndarray): PAPR scores for pristine images (Label 0).
        fake_paprs (list or np.ndarray): PAPR scores for synthetic images (Label 1).

    Returns:
        dict: Summary containing AUC, best accuracy, optimal threshold, and mean scores.
    """
    y_true = np.concatenate([np.zeros(len(real_paprs)), np.ones(len(fake_paprs))])
    y_scores = np.concatenate([real_paprs, fake_paprs])

    # Compute ROC-AUC
    auc = roc_auc_score(y_true, y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Determine optimal operating point (Youden's J statistic)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[best_idx]

    # Calculate classification accuracy at optimal threshold
    y_pred = (y_scores >= optimal_threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)

    return {
        "AUC": float(auc),
        "Accuracy": float(acc * 100.0),
        "Optimal_Threshold": float(optimal_threshold),
        "Mean_PAPR_Real": float(np.mean(real_paprs)),
        "Mean_PAPR_Fake": float(np.mean(fake_paprs)),
        "Std_PAPR_Real": float(np.std(real_paprs)),
        "Std_PAPR_Fake": float(np.std(fake_paprs))
    }
