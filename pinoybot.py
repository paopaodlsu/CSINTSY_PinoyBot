"""
pinoybot.py

PinoyBot: Filipino Code-Switched Language Identifier

This module provides the main tagging function for the PinoyBot project, which identifies the language of each word in a code-switched Filipino-English text. The function is designed to be called with a list of tokens and returns a list of tags ("ENG", "FIL", "CS", or "OTH").

Model training and feature extraction should be implemented in a separate script. The trained model should be saved and loaded here for prediction.
"""

import os
import pickle
from typing import List

from features import extract_features


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
# extract_features is now imported from features.py (see above)


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