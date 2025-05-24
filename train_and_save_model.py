import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load and preprocess
df = pd.read_csv("creditcard.csv")
df = df.drop_duplicates()

normal = df[df['Class'] == 0]
fraud = df[df['Class'] == 1]
normal_sample = normal.sample(n=473, random_state=42)
new_df = pd.concat([normal_sample, fraud], ignore_index=True)

X = new_df.drop(columns=['Class', 'Time'])

y = new_df['Class']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=44)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "rf_fraud_model.pkl")
print("✅ Model saved as rf_fraud_model.pkl")
