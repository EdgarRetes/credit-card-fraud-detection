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

    # Convert all but last column to float
    X = data[:, :-1].astype(float)

    # Convert Class column (remove quotes)
    y = np.array([int(v.strip().strip('"')) for v in data[:, -1]])



    print("Dataset loaded:", X.shape)
    print("Fraud count:", np.sum(y == 1))
    print("Normal count:", np.sum(y == 0))

    fraud_idx = np.where(y == 1)[0]
    normal_idx = np.where(y == 0)[0]

    np.random.shuffle(normal_idx)
    normal_idx = normal_idx[:len(fraud_idx) * 5]

    idx = np.concatenate([fraud_idx, normal_idx])

    X = X[idx]
    y = y[idx]

    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # 4️⃣ Train / Test split
    n = len(X)
    perm = np.random.permutation(n)

    split = int(0.8 * n)
    train_idx = perm[:split]
    test_idx = perm[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print("Train size:", len(X_train))
    print("Test size:", len(X_test))

    rf = RandomForest(n_trees=20, max_depth=1)
    rf.fit(X_train, y_train)

    gb = GradientBoosting(
        n_trees=20,
        learning_rate=0.1,
        max_depth=1
    )
    gb.fit(X_train, y_train)

    x = X_test[0]
    true_label = y_test[0]

    p = ensemble_predict(rf, gb, x)
    phi = shap_ensemble(rf, gb, x)

    print("\n=== Fraud Detection Result ===")
    print("True label:", int(true_label))
    print("Predicted fraud probability:", round(p, 4))

    print("\nSHAP Explanation:")
    for i in sorted(phi.keys()):
        print(f"Feature {i}: {phi[i]:.5f}")

if __name__ == "__main__":
    main()
