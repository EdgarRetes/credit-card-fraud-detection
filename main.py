import numpy as np
from fraud_xai import (
    RandomForest,
    GradientBoosting,
    ensemble_predict,
    shap_ensemble
)

def compute_metrics(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    
    return acc, prec, rec, f1

def run_experiments(rf, gb, X_train, y_train, X_test, y_test, rf_time, gb_time):
    import time
    
    print("\n" + "="*50)
    print("TABLE 1: MODEL COMPARISON")
    print("="*50)
    
    rf_preds = np.array([1 if rf.predict(X_test[i]) >= 0.5 else 0 for i in range(len(X_test))])
    rf_acc, rf_prec, rf_rec, rf_f1 = compute_metrics(y_test, rf_preds)
    
    gb_preds = np.array([1 if gb.predict(X_test[i]) >= 0.5 else 0 for i in range(len(X_test))])
    gb_acc, gb_prec, gb_rec, gb_f1 = compute_metrics(y_test, gb_preds)
    
    ens_probs = np.array([ensemble_predict(rf, gb, X_test[i]) for i in range(len(X_test))])
    ens_preds = (ens_probs >= 0.5).astype(int)
    ens_acc, ens_prec, ens_rec, ens_f1 = compute_metrics(y_test, ens_preds)
    
    print(f"Random Forest:    Acc={rf_acc:.2f}  Prec={rf_prec:.2f}  Rec={rf_rec:.2f}  F1={rf_f1:.2f}")
    print(f"Gradient Boost:   Acc={gb_acc:.2f}  Prec={gb_prec:.2f}  Rec={gb_rec:.2f}  F1={gb_f1:.2f}")
    print(f"Ensemble:         Acc={ens_acc:.2f}  Prec={ens_prec:.2f}  Rec={ens_rec:.2f}  F1={ens_f1:.2f}")
    print(f"\nTraining time: RF={rf_time:.1f}s  GB={gb_time:.1f}s  Total={rf_time+gb_time:.1f}s")
    
    print("\n" + "="*50)
    print("TABLE 2: THRESHOLD ANALYSIS")
    print("="*50)
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        preds_t = (ens_probs >= t).astype(int)
        _, prec_t, rec_t, f1_t = compute_metrics(y_test, preds_t)
        print(f"Threshold {t}: Prec={prec_t:.2f}  Rec={rec_t:.2f}  F1={f1_t:.2f}")
    
    print("\n" + "="*50)
    print("TABLE 3: NOISE ROBUSTNESS")
    print("="*50)
    base_preds = ens_preds.copy()
    for sigma in [0.0, 0.1, 0.2, 0.5]:
        np.random.seed(42)
        X_noisy = X_test + np.random.normal(0, sigma, X_test.shape)
        noisy_probs = np.array([ensemble_predict(rf, gb, X_noisy[i]) for i in range(len(X_noisy))])
        noisy_preds = (noisy_probs >= 0.5).astype(int)
        noisy_acc, _, _, _ = compute_metrics(y_test, noisy_preds)
        stability = np.mean(noisy_preds == base_preds) * 100
        print(f"Noise σ={sigma}: Acc={noisy_acc:.2f}  Stability={stability:.1f}%")
    
    print("\n" + "="*50)
    print("TABLE 4: LATENCY (per sample)")
    print("="*50)
    n_test = min(50, len(X_test))
    
    start = time.time()
    for i in range(n_test):
        rf.predict(X_test[i])
    rf_lat = (time.time() - start) / n_test * 1000
    
    start = time.time()
    for i in range(n_test):
        gb.predict(X_test[i])
    gb_lat = (time.time() - start) / n_test * 1000
    
    start = time.time()
    for i in range(n_test):
        ensemble_predict(rf, gb, X_test[i])
    ens_lat = (time.time() - start) / n_test * 1000
    
    start = time.time()
    for i in range(min(10, n_test)):
        shap_ensemble(rf, gb, X_test[i])
    shap_lat = (time.time() - start) / min(10, n_test) * 1000
    
    print(f"RF prediction:   {rf_lat:.2f} ms")
    print(f"GB prediction:   {gb_lat:.2f} ms")
    print(f"Ensemble:        {ens_lat:.2f} ms")
    print(f"SHAP explain:    {shap_lat:.2f} ms")
    print(f"TOTAL:           {ens_lat + shap_lat:.2f} ms")
    
    print("\n" + "="*50)
    print("SHAP FEATURE IMPORTANCE (Top 10)")
    print("="*50)
    all_shap = {}
    for i in range(min(30, len(X_test))):
        phi = shap_ensemble(rf, gb, X_test[i])
        for k, v in phi.items():
            if k not in all_shap:
                all_shap[k] = []
            all_shap[k].append(abs(v))
    
    mean_shap = [(k, np.mean(v)) for k, v in all_shap.items()]
    mean_shap.sort(key=lambda x: x[1], reverse=True)
    
    for feat, val in mean_shap[:10]:
        name = "Time" if feat == 0 else ("Amount" if feat == 29 else f"V{feat}")
        print(f"{name}: {val:.6f}")

def main():
    import time

    data = np.genfromtxt(
        "creditcard.csv",
        delimiter=",",
        skip_header=1,
        dtype=str
    )

    X = data[:, :-1].astype(float)
    y = np.array([int(v.strip().strip('"')) for v in data[:, -1]])

    fraud_idx = np.where(y == 1)[0]
    normal_idx = np.where(y == 0)[0]

    np.random.shuffle(normal_idx)
    normal_idx = normal_idx[:len(fraud_idx) * 5]

    idx = np.concatenate([fraud_idx, normal_idx])
    X = X[idx]
    y = y[idx]

    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    print("\nDataset info")
    print("Samples:", X.shape[0])
    print("Features:", X.shape[1])
    print("Fraud = 1, Normal = 0")

    n = len(X)
    perm = np.random.permutation(n)
    split = int(0.8 * n)

    X_train = X[perm[:split]]
    y_train = y[perm[:split]]
    X_test = X[perm[split:]]
    y_test = y[perm[split:]]

    print("\nTraining Random Forest...")
    rf = RandomForest(n_trees=30, max_depth=6)
    start = time.time()
    rf.fit(X_train, y_train)
    rf_time = time.time() - start

    print("Training Gradient Boosting...")
    gb = GradientBoosting(n_trees=30, learning_rate=0.1, max_depth=6)
    start = time.time()
    gb.fit(X_train, y_train)
    gb_time = time.time() - start

    run_experiments(rf, gb, X_train, y_train, X_test, y_test, rf_time, gb_time)

    while True:
        cmd = input("\nGenerate random test sample? (y/n): ").strip().lower()
        if cmd != "y":
            break

        i = np.random.randint(len(X_test))
        x = X_test[i]
        true_label = y_test[i]

        p = ensemble_predict(rf, gb, x)
        phi = shap_ensemble(rf, gb, x)

        print("\nSample index:", i)
        print("True label:", int(true_label))
        print("Predicted fraud probability:", round(p, 4))

        print("\nSHAP values:")
        for k in sorted(phi.keys()):
            print(f"Feature {k}: {phi[k]:.8f}")

if __name__ == "__main__":
    main()