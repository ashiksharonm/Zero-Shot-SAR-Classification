"""Classification metrics, baselines, bootstrap CIs and McNemar's test."""
import numpy as np
from scipy import stats
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)


def summary(y, p, n_classes):
    labels = np.arange(n_classes)
    P, R, F1, _ = precision_recall_fscore_support(y, p, labels=labels, average="macro", zero_division=0)
    _, _, wF1, _ = precision_recall_fscore_support(y, p, labels=labels, average="weighted", zero_division=0)
    return {
        "accuracy": accuracy_score(y, p),               # = micro-averaged P/R/F1 for single-label
        "balanced_accuracy": balanced_accuracy_score(y, p),  # = macro-averaged recall
        "macro_precision": P, "macro_recall": R, "macro_f1": F1, "weighted_f1": wF1,
        "n_predicted_classes": int(len(np.unique(p))),
        "n": int(len(y)),
    }


def per_class(y, p, names):
    P, R, F1, S = precision_recall_fscore_support(y, p, labels=np.arange(len(names)), zero_division=0)
    return [{"class": n, "precision": a, "recall": b, "f1": c, "support": int(s)}
            for n, a, b, c, s in zip(names, P, R, F1, S)]


def baselines(y, n_classes):
    """Majority-class and chance baselines on labels y."""
    counts = np.bincount(y, minlength=n_classes)
    prior = counts / counts.sum()
    maj = int(counts.argmax())
    maj_pred = np.full_like(y, maj)
    return {
        "majority": {**summary(y, maj_pred, n_classes), "majority_class": maj},
        "uniform_random_expected_accuracy": 1.0 / n_classes,
        "prior_random_expected_accuracy": float((prior ** 2).sum()),
    }


def bootstrap_ci(y, p, metric="accuracy", n_classes=None, n_boot=1000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if metric == "accuracy":
            vals.append((y[i] == p[i]).mean())
        else:
            vals.append(precision_recall_fscore_support(y[i], p[i], labels=np.arange(n_classes),
                                                        average="macro", zero_division=0)[2])
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def mcnemar(y, p1, p2):
    """Continuity-corrected McNemar test on paired predictions."""
    a = (p1 == y)
    b = (p2 == y)
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    if n01 + n10 == 0:
        return {"chi2": 0.0, "p": 1.0, "only_1_correct": n01, "only_2_correct": n10}
    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return {"chi2": float(chi2), "p": float(stats.chi2.sf(chi2, 1)),
            "only_1_correct": n01, "only_2_correct": n10}


def confusion(y, p, n_classes):
    return confusion_matrix(y, p, labels=np.arange(n_classes))
