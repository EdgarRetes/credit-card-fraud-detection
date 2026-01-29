import numpy as np
from fraud_xai import (
    RandomForest,
    GradientBoosting,
    ensemble_predict,
    shap_ensemble
)

def main():

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

    rf = RandomForest(n_trees=30, max_depth=6)
    rf.fit(X_train, y_train)

    gb = GradientBoosting(
        n_trees=30,
        learning_rate=0.1,
        max_depth=6
    )
    gb.fit(X_train, y_train)

    preds = []
    for i in range(len(X_test)):
        p = ensemble_predict(rf, gb, X_test[i])
        preds.append(1 if p >= 0.5 else 0)

    preds = np.array(preds)
    acc = np.mean(preds == y_test)

    print("\nTest evaluation")
    print("Accuracy:", round(acc, 4))
    print("Total test samples:", len(X_test))

    print("\nFeature indices used:")
    print(list(range(X.shape[1])))

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
