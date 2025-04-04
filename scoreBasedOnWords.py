# This code compares keyword with participant answer.
# 1. Finds one word within the participant answer sentence closest to the keyword based on spelling.
# 2. Checks whether the answer is misspelled and corrects if it is.
# 3. Computes phoneme-based Levenshtein distance with weighted phonetic similarity
# 4. If the distance is relatively small (<0.5), checks if it is a minor spelling error (-ed or -s) and corrects it
# 5. Returns a boolean value of correct/incorrect for each keyword.
# last edited by Jaeeun Lee 2025/03/12
from g2p_en import G2p
from fuzzywuzzy import process #https://pypi.org/project/fuzzywuzzy/
import pandas as pd
import numpy as np
import re
import panphon.featuretable as ft #https://github.com/dmort27/panphon
from spellchecker import SpellChecker

# import nltk
# nltk.download('averaged_perceptron_tagger_eng')

spell = SpellChecker()
g2p = G2p() #initialize the G2P model
ft = ft.FeatureTable() 

############################## FUNCTIONS ##############################
# Converts words to phonemes
def get_phonemes(word):
    return g2p(word)


# Checks if word2 is a likely misspelling of word1 using a spellchecker
def is_misspelling_dict(keyword, resp):
    suggestions = spell.candidates(resp)
    if not suggestions:
        return 999
    return keyword in suggestions


# Computes phonetic similarity based on feature differences
def phoneme_similarity(p1, p2):
    if p1 == p2:
        return 0  # No cost for identical phonemes

    f1 = ft.word_to_vector_list(p1, numeric=True)
    f2 = ft.word_to_vector_list(p2, numeric=True)

    if not f1 or not f2:
        return 1  # Default cost if phoneme is unknown

    # Take the first vector (most common representation)
    f1 = f1[0]
    f2 = f2[0]

    # Compute weighted feature-based cost
    return sum(abs(f1[i] - f2[i]) for i in range(len(f1)))


# Computes phoneme-based Levenshtein distance with weighted phonetic similarity based on the length of phonemes in word 1
def weighted_phoneme_edit_distance(word1, word2):
    p1 = get_phonemes(word1)
    p2 = get_phonemes(word2)

    if not p1 or not p2:
        return float('inf')  # If words can't be converted

    len1, len2 = len(p1), len(p2)
    dp = np.zeros((len1 + 1, len2 + 1))

    # Initialize base cases
    for i in range(len1 + 1):
        dp[i][0] = i  # Deletion cost
    for j in range(len2 + 1):
        dp[0][j] = j  # Insertion cost

    # Fill DP table
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            sub_cost = phoneme_similarity(p1[i-1], p2[j-1])
            dp[i][j] = min(
                dp[i-1][j] + 1,        # Deletion
                dp[i][j-1] + 1,        # Insertion
                dp[i-1][j-1] + sub_cost  # Substitution with weighted cost
            )

    # Compute normalized distance
    raw_distance = dp[len1][len2]
    normalized_distance = raw_distance / len1 if len1 > 0 else float('inf')

    return round(normalized_distance,3)



############################## MAIN CODE ##############################
subDir = "M:/Experiments/Jaeeun/AVBIF/Analysis/fuzzySubs"

for sID in range(1,12):
    D = pd.read_csv("%s/tidySub%d.csv" % (subDir, sID))

    for i, row in D.iterrows():
        row["resp"] = re.sub(r'[^a-zA-Z0-9\s]', '', str(row["resp"]))
        resp = str(row["resp"].lower()).split()
        keyword = row["keyword"].lower()
        compare = process.extractOne(keyword, resp) #how off is the spelling of the response to the keyword

        if compare: #if there is one extracted
            D.at[i,"fuzzy1"] = compare[0] #the fuzzy word
            D.at[i,"fuzzRatio"] = compare[1] #Levenshtein distance [0:100]
            D.at[i,"spellBinary"] = is_misspelling_dict(keyword, compare[0]) #is this a misspelling?
            D.at[i,"wordNum"] = row["wordNum"]
            
            if D.at[i,"spellBinary"]: #if misspelling (including correct word)
                D.at[i,"finalFuzz"] = keyword #fix the misspelling
                if D.at[i,"spellBinary"] == 999: #if can't suggest a correction
                    D.at[i,"finalFuzz"] = compare[0] #fuzzy word (it's going to be marked as incorrect anyway)
            else:
                D.at[i,"finalFuzz"] = compare[0]

            D.at[i,"phoneDistance"] = weighted_phoneme_edit_distance(keyword, D.at[i,"finalFuzz"])

            if D.at[i,"phoneDistance"] <= 0.5:
                D.at[i,"inputBIF"] = 1 - D.at[i,"phoneDistance"]
                if (D.at[i,"fuzzRatio"] >= 80) and (D.at[i,"inputBIF"] != 1):
                    if (keyword in D.at[i,"finalFuzz"]) and ((keyword+"ed" == D.at[i,"finalFuzz"]) or (keyword+"s" == D.at[i,"finalFuzz"])):
                        D.at[i,"inputBIF"] = 1
                    elif (D.at[i,"finalFuzz"] in keyword) and ((D.at[i,"finalFuzz"]+"ed"==keyword) or (D.at[i,"finalFuzz"]+"s"==keyword)):
                        D.at[i,"inputBIF"] = 1
            else:
                D.at[i,"inputBIF"] = 0 

    D.to_csv("%s/fuzzySub%d.csv" % (subDir, sID),index=False)
