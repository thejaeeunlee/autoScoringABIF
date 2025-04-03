# autoScoringABIF

Simple transcript scoring method using fuzzywuzzy and other misc.

## scoreBasedOnWords
From each keyword in the nonsenseSPIN sentence corpus, compare it to the participant's response and return a binary of correct/incorrect.
#### Criteria
- Extract the fuzziest word in the response for each keyword
- Check if it is a spelling error (if it is, correct it)
- If the phonetic difference between the fuzz word and the keyword is small, check if it is just a simple grammatical error (tense, singular/plural)

## scoreBasedOnPhones


Note. Subject 2 is the data that is excluded from the study analysis
