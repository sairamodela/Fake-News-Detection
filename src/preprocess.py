"""
Fake News Detection - Text Preprocessing
Author: Sairam Odela
"""

import re
import string


def clean_text(text: str) -> str:
    """
    Clean and normalise raw news text.
    Steps:
      1. Lowercase
      2. Remove URLs
      3. Remove HTML tags
      4. Remove punctuation & digits
      5. Collapse whitespace
    """
    if not isinstance(text, str):
        return ''

    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)           # remove URLs
    text = re.sub(r'<.*?>', '', text)                      # remove HTML
    text = re.sub(r'\[.*?\]', '', text)                    # remove bracketed text
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)  # punctuation
    text = re.sub(r'\d+', '', text)                        # digits
    text = re.sub(r'\s+', ' ', text).strip()               # whitespace

    return text
