"""
pinoybot.py

PinoyBot: Filipino Code-Switched Language Identifier

This module provides the main tagging function for the PinoyBot project, which identifies the language of each word in a code-switched Filipino-English text. The function is designed to be called with a list of tokens and returns a list of tags ("ENG", "FIL", "CS", or "OTH").

Model training and feature extraction should be implemented in a separate script. The trained model should be saved and loaded here for prediction.
"""

import os
import cloudpickle
from typing import List
<<<<<<< Updated upstream
=======


# -------------------------------------------------------
# Load Saved Model
# -------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLASSIFIER_PATH = os.path.join(BASE_DIR, "classifier.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

with open(CLASSIFIER_PATH, "rb") as f:
    classifier = cloudpickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = cloudpickle.load(f)


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
>>>>>>> Stashed changes

# Main tagging function
def tag_language(tokens: List[str]) -> List[str]:
    """
<<<<<<< Updated upstream
    Tags each token in the input list with its predicted language.
    Args:
        tokens: List of word tokens (strings).
    Returns:
        tags: List of predicted tags ("ENG", "FIL", "CS", or "OTH"), one per token.
=======
    Predict the language tag of every token.

    Parameters
    ----------
    tokens : list[str]

    Returns
    -------
    list[str]
>>>>>>> Stashed changes
    """
    # 1. Load your trained model from disk (e.g., using pickle or joblib)
    #    Example: with open('trained_model.pkl', 'rb') as f: model = pickle.load(f)
    #    (Replace with your actual model loading code)

<<<<<<< Updated upstream
    # 2. Extract features from the input tokens to create the feature matrix
    #    Example: features = ... (your feature extraction logic here)
=======
    if not tokens:
        return []
>>>>>>> Stashed changes

    # 3. Use the model to predict the tags for each token
    #    Example: predicted = model.predict(features)

    # 4. Convert the predictions to a list of strings ("ENG", "FIL", or "OTH")
    #    Example: tags = [str(tag) for tag in predicted]

    # 5. Return the list of tags
    #    return tags

<<<<<<< Updated upstream
    # You can define other functions, import new libraries, or add other Python files as needed, as long as
    # the tag_language function is retained and correctly accomplishes the expected task.

    # Currently, the bot just tags every token as FIL. Replace this with your more intelligent predictions.
    return ['FIL' for i in tokens]

if __name__ == "__main__":
    # Example usage
    example_tokens = ["Love", "kita", "."]
    print("Tokens:", example_tokens)
    tags = tag_language(example_tokens)
=======
    return predictions.tolist()
>>>>>>> Stashed changes
