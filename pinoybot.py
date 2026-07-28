"""
pinoybot.py

PinoyBot: Filipino Code-Switched Language Identifier

This module provides the main tagging function for the PinoyBot project, which identifies the language of each word in a code-switched Filipino-English text. The function is designed to be called with a list of tokens and returns a list of tags ("ENG", "FIL", "CS", or "OTH").

Model training and feature extraction should be implemented in a separate script. The trained model should be saved and loaded here for prediction.
"""

import os
import pickle
from typing import List


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

        "ends_vowel": (
            len(word) > 0 and
            word[-1].lower() in "aeiou"
        ),

        "contains_ng": "ng" in word.lower(),
        "contains_mag": "mag" in word.lower(),
        "contains_nag": "nag" in word.lower(),
        "contains_um": "um" in word.lower(),
        "contains_in": "in" in word.lower()
    }


# -------------------------------------------------------
# Main Prediction Function
# -------------------------------------------------------

def tag_language(tokens: List[str]) -> List[str]:
    """
    Predict the language tag of every token.

    Parameters
    ----------
    tokens : list[str]

    Returns
    -------
    list[str]
    """

    if not tokens:
        return []

    features = [extract_features(token) for token in tokens]

    X = vectorizer.transform(features)

    predictions = classifier.predict(X)

    return predictions.tolist()