
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

columns = [
    "Pregnancies", "Glucose", "BloodPressure",
    "SkinThickness", "Insulin", "BMI",
    "DiabetesPedigreeFunction", "Age", "Outcome"
]

df = pd.read_csv("diabetes.csv", names=columns)

cols = ["Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI"]

df[cols] = df[cols].replace(0, float("nan"))
df[cols] = df[cols].fillna(df[cols].median())

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model1 = LogisticRegression(max_iter=1000)
model1.fit(X_train, y_train)
pred1 = model1.predict(X_test)

print("WITHOUT FEATURE SCALING")
print("Accuracy:", accuracy_score(y_test, pred1))
print("Precision:", precision_score(y_test, pred1, zero_division=0))
print("Recall:", recall_score(y_test, pred1, zero_division=0))
print("F1-score:", f1_score(y_test, pred1, zero_division=0))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model2 = LogisticRegression(max_iter=1000)
model2.fit(X_train_scaled, y_train)
pred2 = model2.predict(X_test_scaled)

print("\nWITH FEATURE SCALING")
print("Accuracy:", accuracy_score(y_test, pred2))
print("Precision:", precision_score(y_test, pred2, zero_division=0))
print("Recall:", recall_score(y_test, pred2, zero_division=0))
print("F1-score:", f1_score(y_test, pred2, zero_division=0))
