"""
ASL Sign Language Recognition - Model Training
Improvements: data augmentation, better RF params, XGBoost
"""

import os, pickle, json
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils import resample



import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── CONFIG ───────────────────────────────────────────────
DATA_CSV     = "landmarks/landmarks_fixed.csv"
LABEL_COL    = "label"
MODELS_DIR   = "output/models"
PLOTS_DIR    = "output/plots"
REPORT_FILE  = "output/comparison_report.json"
TEST_SIZE    = 0.2
RANDOM_STATE = 42

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

# ── AUGMENTATION ─────────────────────────────────────────
def augment_data(X, y, noise_std=0.008, copies=2):
    """Add Gaussian noise copies — expands dataset without overfitting."""
    X_aug, y_aug = [X], [y]
    rng = np.random.default_rng(RANDOM_STATE)
    for _ in range(copies):
        noise = rng.normal(0, noise_std, X.shape).astype(np.float32)
        X_aug.append(X + noise)
        y_aug.append(y)
    return np.vstack(X_aug), np.concatenate(y_aug)

def oversample_minority(X, y, min_samples=500):
    """Oversample classes with fewer than min_samples examples."""
    classes, counts = np.unique(y, return_counts=True)
    X_list, y_list = [X], [y]
    for cls, cnt in zip(classes, counts):
        if cnt < min_samples:
            idx = np.where(y == cls)[0]
            X_cls = X[idx]
            y_cls = y[idx]
            n_needed = min_samples - cnt
            X_resampled = resample(X_cls, n_samples=n_needed,
                                   random_state=RANDOM_STATE, replace=True)
            y_resampled = np.full(n_needed, cls)
            X_list.append(X_resampled)
            y_list.append(y_resampled)
    return np.vstack(X_list), np.concatenate(y_list)

# ── 1. LOAD DATA ─────────────────────────────────────────
def load_data():
    print("\n[1/6] Loading data...")
    df = pd.read_csv(DATA_CSV)
    print(f"  -> {len(df)} samples, {df[LABEL_COL].nunique()} classes")

    drop_cols = [LABEL_COL] + [c for c in ["file"] if c in df.columns]
    X = df.drop(columns=drop_cols).values.astype(np.float32)
    le = LabelEncoder()
    y  = le.fit_transform(df[LABEL_COL].values)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Oversample minority classes BEFORE scaling (on train only)
    print("  -> Oversampling minority classes...")
    X_train, y_train = oversample_minority(X_train, y_train, min_samples=400)

    # Augment with noise
    print("  -> Augmenting with noise...")
    X_train, y_train = augment_data(X_train, y_train, noise_std=0.008, copies=2)

    # Scale AFTER augmentation
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"  -> Train: {len(X_train)}  |  Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, le, scaler

# ── 2. RANDOM FOREST ─────────────────────────────────────
def train_random_forest(X_train, X_test, y_train, y_test, le):
    print("\n[2/6] Training Random Forest...")
    param_grid = {
        "n_estimators":      [200, 300],
        "max_depth":         [None, 30],
        "min_samples_split": [2, 3],
        "max_features":      ["sqrt", "log2"],
    }

    rf = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        param_grid, cv=3, scoring="f1_macro", n_jobs=-1
    )
    rf.fit(X_train, y_train)
    best = rf.best_estimator_

    y_pred = best.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro")
    print(f"  -> Best params: {rf.best_params_}")
    print(f"  -> Accuracy: {acc:.4f}  |  F1: {f1:.4f}")

    path = os.path.join(MODELS_DIR, "random_forest.pkl")
    with open(path, "wb") as f: pickle.dump(best, f)
    print(f"  -> Saved -> {path}")
    _confusion_matrix(y_test, y_pred, le, "RandomForest")

    return {"model": "RandomForest", "accuracy": round(acc,4), "f1_macro": round(f1,4),
            "best_params": rf.best_params_, "model_path": path,
            "report": classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)}

# ── 3. MLP ───────────────────────────────────────────────
def train_mlp(X_train, X_test, y_train, y_test, le):
    print("\n[3/6] Training MLP...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128, 64),
        activation="relu", solver="adam",
        alpha=1e-3, learning_rate="adaptive",
        learning_rate_init=1e-3, max_iter=300,
        early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=15, random_state=RANDOM_STATE, verbose=False
    )
    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro")
    print(f"  -> Stopped at iter: {mlp.n_iter_}")
    print(f"  -> Accuracy: {acc:.4f}  |  F1: {f1:.4f}")

    path = os.path.join(MODELS_DIR, "sklearn_mlp.pkl")
    with open(path, "wb") as f: pickle.dump(mlp, f)
    print(f"  -> Saved -> {path}")
    _confusion_matrix(y_test, y_pred, le, "MLP")
    _training_curve(mlp)

    return {"model": "SklearnMLP", "accuracy": round(acc,4), "f1_macro": round(f1,4),
            "best_params": {"layers": "512->256->128->64", "solver": "adam", "iters": mlp.n_iter_},
            "model_path": path,
            "report": classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)}

# ── 4. COMPARE & SAVE ────────────────────────────────────
def compare_and_save(results, le, scaler):
    print("\n[5/6] Comparing...")
    for r in results:
        print(f"  {r['model']:15} -> Acc: {r['accuracy']:.4f}  F1: {r['f1_macro']:.4f}")

    winner = max(results, key=lambda r: r["f1_macro"])
    print(f"  Winner: {winner['model']}")

    le_path = os.path.join(MODELS_DIR, "label_encoder.pkl")
    sc_path = os.path.join(MODELS_DIR, "scaler.pkl")
    with open(le_path, "wb") as f: pickle.dump(le, f)
    with open(sc_path, "wb") as f: pickle.dump(scaler, f)

    report = {
        "generated_at": datetime.now().isoformat(),
        "winner": winner["model"],
        **{r["model"].lower().replace(" ", "_"): r for r in results},
        "artifacts_for_person3": {
            "best_model":    winner["model_path"],
            "label_encoder": le_path,
            "scaler":        sc_path,
            "note": "Person 3: apply scaler.transform() on landmarks before inference."
        }
    }
    with open(REPORT_FILE, "w") as f: json.dump(report, f, indent=2)
    print(f"  -> Report saved -> {REPORT_FILE}")
    _comparison_bar(results)
    return report

# ── PLOT HELPERS ─────────────────────────────────────────
def _confusion_matrix(y_true, y_pred, le, name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_title(f"Confusion Matrix - {name}")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    path = os.path.join(PLOTS_DIR, f"cm_{name.lower()}.png")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  -> Confusion matrix -> {path}")

def _training_curve(mlp):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(mlp.loss_curve_, label="train loss")
    if mlp.validation_scores_ is not None:
        ax2 = ax.twinx()
        ax2.plot(mlp.validation_scores_, color="orange", label="val accuracy")
        ax2.set_ylabel("Val accuracy"); ax2.legend(loc="lower right")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Loss")
    ax.set_title("MLP Training Curve"); ax.legend(loc="upper right")
    path = os.path.join(PLOTS_DIR, "mlp_training_curve.png")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  -> Training curve -> {path}")

def _comparison_bar(results):
    names = [r["model"] for r in results]
    accs  = [r["accuracy"] for r in results]
    f1s   = [r["f1_macro"] for r in results]
    x = np.arange(len(names)); w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x-w/2, accs, w, label="Accuracy", color="#4C9BE8")
    ax.bar(x+w/2, f1s,  w, label="F1 Macro", color="#F4845F")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05); ax.set_title("Model Comparison"); ax.legend()
    path = os.path.join(PLOTS_DIR, "model_comparison.png")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  -> Comparison chart -> {path}")

# ── MAIN ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  ASL - Model Training Pipeline (Improved)")
    print("=" * 55)

    X_train, X_test, y_train, y_test, le, scaler = load_data()

    results = []
    results.append(train_random_forest(X_train, X_test, y_train, y_test, le))
    results.append(train_mlp(X_train, X_test, y_train, y_test, le))

    report = compare_and_save(results, le, scaler)

    print("\n[6/6] Done! Files ready for Person 3:")
    for k, v in report["artifacts_for_person3"].items():
        if k != "note": print(f"  {k:15} -> {v}")
    print(f"\n  Note: {report['artifacts_for_person3']['note']}")
    print("\nAll done!\n")