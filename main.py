import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
joblib.dump(model, "knn_model.joblib")
df = pd.read_csv("parkinsons.csv")

features = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Jitter(%)",
            "MDVP:Shimmer", "HNR", "RPDE", "DFA"]
X = df[features]
y = df["status"]

scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print("דיוק המודל:", accuracy)

joblib.dump(model, "knn_model.joblib")
