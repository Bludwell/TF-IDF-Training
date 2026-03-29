# TF-IDF Multi-Label Klassifikation (Gesundheitstexte)

## Übersicht

Dieses Repository enthält eine Trainingspipeline für ein **klassisches Machine-Learning-Modell** zur Analyse kurzer, deutschsprachiger Freitexte.

Das Modell basiert auf:

* **TF-IDF-Vektorisierung**
* **Logistischer Regression (One-vs-Rest)**

Ziel ist die **Multi-Label-Klassifikation** der folgenden Kategorien:

* Schlaf
* Stress
* Bewegung
* Ernährung

---

## Funktionsweise

Die Pipeline besteht aus:

1. Laden eines CSV-Datensatzes (`data.csv`)
2. Train/Test-Split (80/20)
3. Text-Vektorisierung mit TF-IDF
4. Training eines Multi-Label-Klassifikators
5. Evaluation mit Classification Report
6. Speicherung des Modells (`.joblib`)

---

## Datensatz

Erwartetes Format:

| text                      | schlaf | stress | bewegung | ernaehrung |
| ------------------------- | ------ | ------ | -------- | ---------- |
| "Ich schlafe schlecht..." | 1      | 1      | 0        | 0          |

* `text`: Freitext
* Labels: 0 oder 1 (Multi-Label möglich)

---

## Installation
Benötigte Libraries:

* scikit-learn
* pandas
* numpy
* joblib

---

## Training starten

```bash id="c1n09g"
python train.py
```

---

## Modell

* TF-IDF mit:

  * Unigrammen + Bigrammen (`ngram_range=(1,2)`)
  * max. 5000 Features
* Klassifikator:

  * Logistic Regression
  * One-vs-Rest Strategie
* Multi-Label Output

---

## Output

* `model.joblib` → trainiertes Modell
* `evaluation_results.csv` → Metriken (Precision, Recall, F1)

---

## Evaluation

* Standard: Threshold = 0.5
* Zusätzlich getestet:

  * Threshold 0.35 (höherer Recall)
  * Individuelle Thresholds pro Label

Dies ermöglicht eine bessere Balance zwischen:

* Precision
* Recall

---

## Beispiel

Input:

```id="9b2v4m"
ich esse und schlafe zu wenig
```

Output (Beispiel):

```id="o1kv7r"
Schlaf ✓
Stress ✓
Bewegung ✗
Ernährung ✓
```

---

## Hinweis

Dieses Modell dient als **Baseline-Ansatz** und wurde im Rahmen einer prototypischen Anwendung entwickelt.
