"""
train_model.py

Trains PinoyBot's language identification model.

Input:
    67.xlsx

Output:
    classifier.pkl
    vectorizer.pkl
"""

import cloudpickle
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support
)
from sklearn.model_selection import train_test_split


# -------------------------------------------------------
# Feature Extraction
# -------------------------------------------------------

def extract_features(word):

    word = str(word)

    vowels = "aeiouAEIOU"

    return {

        "word": word,
        "lower": word.lower(),

        "length": len(word),

        "is_upper": word.isupper(),
        "is_lower": word.islower(),
        "is_title": word.istitle(),

        "is_digit": word.isdigit(),
        "has_digit": any(c.isdigit() for c in word),

        "has_hyphen": "-" in word,
        "has_apostrophe": "'" in word,

        "prefix1": word[:1],
        "prefix2": word[:2],
        "prefix3": word[:3],

        "suffix1": word[-1:],
        "suffix2": word[-2:],
        "suffix3": word[-3:],

        "num_vowels": sum(c in vowels for c in word),
        "num_upper": sum(c.isupper() for c in word),
        "num_digits": sum(c.isdigit() for c in word),

        "starts_capital": word[:1].isupper(),

        "ends_vowel":
            len(word) > 0 and word[-1].lower() in "aeiou",

        "contains_ng": "ng" in word.lower(),
        "contains_mag": "mag" in word.lower(),
        "contains_nag": "nag" in word.lower(),
        "contains_um": "um" in word.lower(),
        "contains_in": "in" in word.lower()
    }


# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------

print("Loading dataset...")

df = pd.read_excel("67.xlsx")

df = df.dropna(subset=["word", "label"])

# Clean labels
df["label"] = (
    df["label"]
    .astype(str)
    .str.strip()
    .str.upper()
)

# Fix common annotation mistakes
df["label"] = df["label"].replace({
    "FII": "FIL",
    "FILL": "FIL",
    "FIL.": "FIL",
    "O": "OTH",
    "OTJ": "OTH",
    "EOTH": "OTH",
})

VALID = ["ENG", "FIL", "CS", "OTH"]

df = df[df["label"].isin(VALID)]

print(df["label"].value_counts())


# -------------------------------------------------------
# Build Feature Matrix
# -------------------------------------------------------

X = [extract_features(word) for word in df["word"]]

y = df["label"]


# -------------------------------------------------------
# Vectorize
# -------------------------------------------------------

vectorizer = DictVectorizer()

X = vectorizer.fit_transform(X)


# -------------------------------------------------------
# Train / Validation / Test
# -------------------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


print("Training:", len(y_train))
print("Validation:", len(y_valid))
print("Testing:", len(y_test))


# -------------------------------------------------------
# Train Model
# -------------------------------------------------------

classifier = RandomForestClassifier(
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)

classifier.fit(X_train, y_train)


# -------------------------------------------------------
# Validation Results
# -------------------------------------------------------

print("\nValidation Results")

valid_pred = classifier.predict(X_valid)

print("Accuracy:",
      accuracy_score(y_valid, valid_pred))

precision, recall, f1, _ = precision_recall_fscore_support(
    y_valid,
    valid_pred,
    average="weighted"
)

print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)

print(classification_report(y_valid, valid_pred))


# -------------------------------------------------------
# Test Results
# -------------------------------------------------------

print("\nTest Results")

test_pred = classifier.predict(X_test)

print("Accuracy:",
      accuracy_score(y_test, test_pred))

precision, recall, f1, _ = precision_recall_fscore_support(
    y_test,
    test_pred,
    average="weighted"
)

print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)

print(classification_report(y_test, test_pred))


# -------------------------------------------------------
# Save Model
# -------------------------------------------------------

with open("classifier.pkl", "wb") as f:
    cloudpickle.dump(classifier, f)

with open("vectorizer.pkl", "wb") as f:
    cloudpickle.dump(vectorizer, f)

print("\nSaved:")
print("classifier.pkl")
print("vectorizer.pkl")