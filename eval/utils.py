"Quick script for evaluating agent's tool use"
import dotenv
from typing import List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from together import Together

dotenv.load_dotenv()

#==================================================================================#
#===== Get variables and set global variables =====================================#
#==================================================================================#

client      = Together()

def _base_levenshtein(seq1, seq2):
    n, m = len(seq1), len(seq2)
    dp = np.zeros((n + 1, m + 1), dtype=int)

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                cost = 0  # Same base label
            else:
                cost = 1  # Different base labels
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # Deletion
                dp[i][j - 1] + 1,      # Insertion
                dp[i - 1][j - 1] + cost  # Substitution
            )

    return dp[n][m]

def calc_tcs_levenstein(expected_tcs: List[str], actual_tcs:List[str]):
    return _base_levenshtein(expected_tcs, actual_tcs)

def _get_embedding(text: str, model="text-embedding-ada-002"):
    response = client.embeddings.create(
        model="BAAI/bge-large-en-v1.5",
        input=text
        )

    return response.data[0].embedding

def recall_by_embedding_similarity(list_predict, list_actual, threshold=0.85):

    if not list_actual:
        return 0.0  # Avoid division by zero

    if list_actual == None or list_actual==[""]: list_actual = ["None"]
    if list_predict == None or list_predict==[""]: list_predict = ["None"]

    # Get embeddings
    embeddings_pred = [_get_embedding(pred) for pred in list_predict]
    embeddings_actual = [_get_embedding(act) for act in list_actual]

    # Calculate true positives
    true_positive = 0
    for actual_emb in embeddings_actual:
        max_sim = max(
            cosine_similarity([actual_emb], [pred_emb])[0][0]
            for pred_emb in embeddings_pred
        )
        if max_sim > threshold:
            true_positive += 1

    recall = true_positive / len(list_actual)
    return recall

def calc_tools_acc(
    tools_called  : list[str],
    tools_expected: list[str]
)->float:
    TP = len(set(tools_called).intersection(set(tools_expected)))
    FN = len(set(tools_expected).difference(set(tools_called)))
    if TP+FN == 0:
        recall = 1
    else:
        recall = float(TP/(TP+FN))
    return recall
