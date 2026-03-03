import numpy as np

def auc(fpr, tpr):
    """
    Compute AUC (Area Under ROC Curve) using trapezoidal rule.
    """
    # Write code here
    fpr = np.asarray(fpr, dtype = float)
    tpr = np.asarray(tpr, dtype = float)

    if len(fpr) < 2:
        return 0.0

    auc = 0.0
    for i in range(len(fpr) - 1):
        width = fpr[i + 1] - fpr[i]
        height = (tpr[i] + tpr[i + 1]) / 2.0
        auc += width * height

    return float(max(0.0, min(1.0, auc)))
    pass