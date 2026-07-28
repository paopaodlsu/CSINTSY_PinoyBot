"""
train_model.py

Trains PinoyBot's language identification model.

Input:
    67.xlsx

Output:
    classifier.pkl
    vectorizer.pkl
"""

import pickle
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)
from sklearn.model_selection import train_test_split


# -------------------------------------------------------
# Feature Extraction
# -------------------------------------------------------

def extract_features(word):
    """
    Extracts handcrafted features from a single token.
    """

    word = str(word)

    vowels = "aeiouAEIOU"

    return {

        # Original forms
        "word": word,
        "lower": word.lower(),

        # Length
        "length": len(word),

        # Capitalization
        "is_upper": word.isupper(),
        "is_lower": word.islower(),
        "is_title": word.istitle(),

        # Digits
        "is_digit": word.isdigit(),
        "has_digit": any(c.isdigit() for c in word),

        # Special characters
        "has_hyphen": "-" in word,
        "has_apostrophe": "'" in word,

        # Prefixes
        "prefix1": word[:1],
        "prefix2": word[:2],
        "prefix3": word[:3],

        # Suffixes
        "suffix1": word[-1:],
        "suffix2": word[-2:],
        "suffix3": word[-3:],

        # Character statistics
        "num_vowels": sum(c in vowels for c in word),
        "num_upper": sum(c.isupper() for c in word),
        "num_digits": sum(c.isdigit() for c in word),

        "starts_capital": word[:1].isupper(),

        "ends_vowel":
            len(word) > 0 and word[-1].lower() in "aeiou",

        "contains_ng":
            "ng" in word.lower(),

        "contains_mag":
            "mag" in word.lower(),

        "contains_nag":
            "nag" in word.lower(),

        "contains_um":
            "um" in word.lower(),

        "contains_in":
            "in" in word.lower()
    }


# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------

print("Loading dataset...")

df = pd.read_excel("67.xlsx")

# Remove rows with missing values
df = df.dropna(subset=["word", "label"])

# Convert labels to strings
df["label"] = df["label"].astype(str)

# Remove leading/trailing whitespace
df["label"] = df["label"].str.strip()

# Convert to uppercase
df["label"] = df["label"].str.upper()

# Fix common annotation mistakes
df["label"] = df["label"].replace({
    "FII": "FIL",
    "FII.": "FIL",
    "FIL.": "FIL",
    "FILL": "FIL",
    "O": "OTH",
    "OTJ": "OTH",
    "EOTH": "OTH",
})

# Keep only valid labels
VALID_LABELS = ["ENG", "FIL", "CS", "OTH"]

invalid = df[~df["label"].isin(VALID_LABELS)]

if len(invalid) > 0:
    print("\nFound invalid labels:")
    print(invalid["label"].value_counts())
    print()

df = df[df["label"].isin(VALID_LABELS)]

print(f"Total tokens: {len(df)}")

print("\nLabel counts:")
print(df["label"].value_counts())


# -------------------------------------------------------
# Build Features
# -------------------------------------------------------

print("Extracting features...")

X = [extract_features(word) for word in df["word"]]

y = df["label"]


# -------------------------------------------------------
# Vectorize
# -------------------------------------------------------

vectorizer = DictVectorizer(sparse=True)

X = vectorizer.fit_transform(X)


# -------------------------------------------------------
# 70-15-15 Split
# -------------------------------------------------------

try:

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

except ValueError:

    print("\nWARNING:")
    print("Some classes are too small for stratified splitting.")
    print("Using random split instead.\n")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42
    )

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42
    )

print("Training:", len(y_train))
print("Validation:", len(y_valid))
print("Testing:", len(y_test))


print()

print("Training:", len(y_train))

print("Validation:", len(y_valid))

print("Testing:", len(y_test))

print()


# -------------------------------------------------------
# Train Classifier
# -------------------------------------------------------

print("Training Random Forest...")

classifier = RandomForestClassifier(

    n_estimators=200,

    max_depth=None,

    random_state=42,

    n_jobs=-1
)

classifier.fit(X_train, y_train)


# -------------------------------------------------------
# Validation Evaluation
# -------------------------------------------------------

print("=" * 60)

print("VALIDATION RESULTS")

print("=" * 60)

valid_pred = classifier.predict(X_valid)

accuracy = accuracy_score(y_valid, valid_pred)

precision, recall, f1, _ = precision_recall_fscore_support(

    y_valid,

    valid_pred,

    average="weighted"
)

print(f"Accuracy : {accuracy:.4f}")

print(f"Precision: {precision:.4f}")

print(f"Recall   : {recall:.4f}")

print(f"F1 Score : {f1:.4f}")

print()

print(classification_report(y_valid, valid_pred))


# -------------------------------------------------------
# Test Evaluation
# -------------------------------------------------------

print("=" * 60)

print("TEST RESULTS")

print("=" * 60)

test_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, test_pred)

precision, recall, f1, _ = precision_recall_fscore_support(

    y_test,

    test_pred,

    average="weighted"
)

print(f"Accuracy : {accuracy:.4f}")

print(f"Precision: {precision:.4f}")

print(f"Recall   : {recall:.4f}")

print(f"F1 Score : {f1:.4f}")

print()

print(classification_report(y_test, test_pred))


# -------------------------------------------------------
# Save Model
# -------------------------------------------------------

print()

print("Saving model...")

with open("classifier.pkl", "wb") as f:
    pickle.dump(classifier, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Done!")

print()

print("Generated files:")

print("classifier.pkl")

print("vectorizer.pkl")