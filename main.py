import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# --------------------------------------------------
# 1) Daten laden
# --------------------------------------------------
df = pd.read_csv("data.csv")
# Erwartetes Format der CSV:
# text,schlaf,stress,bewegung,ernaehrung
# "Ich bin total erschöpft und schlafe schlecht",1,1,0,0

X = df["text"]
y = df[["schlaf", "stress", "bewegung", "ernaehrung"]]  # 2D-Matrix mit 0/1

# --------------------------------------------------
# 2) Train/Test-Split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# 3) Pipeline definieren
# --------------------------------------------------
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),   # Einzel- und Bigramme (z.B. "schlechter schlaf")
        min_df=2,             # ignoriert sehr seltene Begriffe
        max_features=5000     # begrenzt Vokabulargröße
    )),
    ("clf", OneVsRestClassifier(
        LogisticRegression(max_iter=1000, C=1.0)
    ))
])

# --------------------------------------------------
# 4) Training
# --------------------------------------------------
pipeline.fit(X_train, y_train)

# --------------------------------------------------
# 5) Evaluation
# --------------------------------------------------
y_pred = pipeline.predict(X_test)

labels = ["schlaf", "stress", "bewegung", "ernaehrung"]
print(classification_report(y_test, y_pred, target_names=labels))

# Ergebnisse als CSV speichern (für Anhang/Arbeit)
report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
pd.DataFrame(report).transpose().to_csv("evaluation_results.csv")

# --------------------------------------------------
# 6) Modell speichern
# --------------------------------------------------
joblib.dump(pipeline, "model.joblib")
print("Modell gespeichert als model.joblib")

# --------------------------------------------------
# 7) Kurze Smoke-Tests (optional aber empfehlenswert)
# --------------------------------------------------
test_texts = [
    "ich esse und schlafe zu wenig",
    "Ich stehe unter extremem Druck bei der Arbeit",
    "Ich bewege mich kaum und esse viel Fast Food",
]

loaded_model = joblib.load("model.joblib")
predictions = loaded_model.predict(test_texts)
probabilities = loaded_model.predict_proba(test_texts)

for text, pred, prob in zip(test_texts, predictions, probabilities):
    print(f"\nText: {text}")
    for label, p, score in zip(labels, pred, prob):
        print(f"  {label}: {'✓' if p == 1 else '✗'}  (Score: {score:.2f})")

print(classification_report(y_test, y_pred, target_names=labels, zero_division=0))

proba = pipeline.predict_proba(X_test)
y_pred_low = (proba >= 0.35).astype(int)

print("\n--- Threshold 0.35")
print(classification_report(y_test, y_pred_low, target_names=labels, zero_division=0))

# Verschiedene Thresholds pro Label
thresholds = {
    "schlaf":     0.50,  # Precision bereits gut, kein Grund zu senken
    "stress":     0.45,  # leicht senken für besseren Recall
    "bewegung":   0.35,  # aggressiver senken, Recall zu niedrig
    "ernaehrung": 0.40,  # Mittelweg
}

proba = pipeline.predict_proba(X_test)
threshold_list = [thresholds[l] for l in labels]
y_pred_custom = (proba >= threshold_list).astype(int)

print("custom thresholds")
print(classification_report(y_test, y_pred_custom, target_names=labels, zero_division=0))