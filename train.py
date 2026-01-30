import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model():
    np.random.seed(42)
    X = np.random.randn(1000, 10)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model accuracy: {accuracy:.4f}")
    return model

if __name__ == "__main__":
    train_model()
