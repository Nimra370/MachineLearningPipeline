from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from data.load_data import load_and_preprocess_data

def train_and_save_model():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Model accuracy: {acc:.2f}")

    joblib.dump(model, 'models/model.pkl')
    return acc
