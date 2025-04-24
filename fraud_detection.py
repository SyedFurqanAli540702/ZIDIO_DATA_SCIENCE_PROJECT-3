import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load historical dataset (simulated for example)
def load_data():
    # Simulating data; replace with actual dataset path
    data = pd.read_csv('historical_transactions.csv')
    return data

# Train model
def train_model(data):
    X = data.drop(['is_fraud'], axis=1)
    y = data['is_fraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))
    joblib.dump(model, 'fraud_model.pkl')

# Load and use model
def predict_transaction(input_data):
    model = joblib.load('fraud_model.pkl')
    prediction = model.predict([input_data])
    return prediction[0]

if __name__ == "__main__":
    data = load_data()
    train_model(data)
