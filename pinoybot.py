"""
pinoybot.py

PinoyBot: Filipino Code-Switched Language Identifier

This module provides the main tagging function for the PinoyBot project, which identifies the language of each word in a code-switched Filipino-English text. The function is designed to be called with a list of tokens and returns a list of tags ("ENG", "FIL", "CS", or "OTH").

Model training and feature extraction should be implemented in a separate script. The trained model should be saved and loaded here for prediction.
"""

import os
import pickle
from typing import List
import pandas as pd


# -------------------------------------------------------
# Load Saved Model
# -------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLASSIFIER_PATH = os.path.join(BASE_DIR, "classifier.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

with open(CLASSIFIER_PATH, "rb") as f:
    classifier = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)


# -------------------------------------------------------
# Feature Extraction
# -------------------------------------------------------

def extract_features(word):
    """
    Extracts the same features used during training.
    """

    word = str(word)

    vowels = "aeiouAEIOU"

    return {

        # Original word
        "word": word,
        "lower": word.lower(),

        # Length
        "length": len(word),

        # Capitalization
        "is_upper": word.isupper(),
        "is_lower": word.islower(),
        "is_title": word.istitle(),

        # Numbers
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

        # Filipino-related patterns
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
# Main Prediction Function
# -------------------------------------------------------

def tag_language(tokens: List[str]) -> List[str]:
    """
    Predicts the language tag for every token.

    Parameters
    ----------
    tokens : List[str]

    Returns
    -------
    List[str]

    Possible labels:
        ENG
        FIL
        CS
        OTH
    """

    if len(tokens) == 0:
        return []

    features = [extract_features(token) for token in tokens]

    X = vectorizer.transform(features)

    predictions = classifier.predict(X)

    return predictions.tolist()


# -------------------------------------------------------
# Example Usage
# -------------------------------------------------------

if __name__ == "__main__":

    # Read the raw tokens
    df = pd.read_excel("raw_tokens.xlsx")

    # Get all words
    tokens = df["word"].fillna("").astype(str).tolist()

    # Predict labels
    predictions = tag_language(tokens)

    # Save predictions
    df["predicted_label"] = predictions

    output_file = "predictions.xlsx"

    df.to_excel(output_file, index=False)

    print(f"Predictions saved to {output_file}")

    print(df.head())