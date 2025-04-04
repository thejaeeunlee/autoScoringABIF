# autoScoringABIF

Simple transcript scoring method using fuzzywuzzy and other misc.

## scoreBasedOnWords.py
From each keyword in the nonsenseSPIN sentence corpus, compare it to the participant's response and return a binary of correct/incorrect.
#### Procedure
- Finds one word within the participant answer sentence closest to the keyword based on spelling.
- Checks whether the answer is misspelled and corrects if it is.
- Computes phoneme-based Levenshtein distance with weighted phonetic similarity.
- If the distance is relatively small (<0.5), checks if it is a minor spelling error (-ed or -s) and corrects it.
- Returns a boolean value of correct/incorrect for each keyword.

## scoreBasedOnPhones.py
From each keyword in the nonsenseSPIN sentence corpus, compare it to the participant's response and return a binary of correct/incorrect of each phoneme in the word.
#### Procedure
- Finds one word within the participant answer sentence closest to the keyword based on spelling.
- Checks whether the answer is misspelled and corrects if it is.
- Computes phoneme-based Levenshtein distance with weighted phonetic similarity.
- If the distance is relatively small (<0.5), compares every phoneme in both keyword and fuzz word.
- Returns a boolean value of correct/incorrect for each phoneme.

Note. Subject 2 is the data that is excluded from the study analysis
