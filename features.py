"""
features.py

Shared feature extraction logic for PinoyBot.

This module is imported by BOTH train_model.py (to build the training
feature matrix) and pinoybot.py (to build features for new input at
prediction time). Keeping the logic in one place guarantees that training
and prediction always use identical features.
"""


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
