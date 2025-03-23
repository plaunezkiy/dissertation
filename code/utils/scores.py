import numpy as np
from typing import List
from utils.preprocessing import preprocess_text
from rouge import Rouge


rouge = Rouge()


def get_rouge_score_for_answers(actual_answers, model_answers):
        """
        Calculate ROUGE score between every actual and every model answer.
        Pick the highest one and return as final.
        """
        max_r = 0
        # print(model_answers)
        for a in actual_answers:
            for m in model_answers:
                if m and a:
                    scores = rouge.get_scores(m[:300], a)[0]
                    if scores["rouge-l"]["r"] > max_r:
                        max_r = scores["rouge-l"]["r"]
        return max_r


def calculate_em_accuracy(actual: List[str], model: List[str]) -> float:
    """
    Calculate the Exact Match accuracy of the model predictions.
    Returns the maximum exact match.
    """
    for m in model:
        for a in actual:
            if preprocess_text(m) == preprocess_text(a):
            # if a == m:
                return 1
    return 0

def f1_score(ground: str, prediction: str) -> float:
    """
    Calculate the F1 score of the model predictions.
    beta=1
    beta > 1 -> more weight to recall
    beta < 1 -> more weight to precision
    """
    ground_tokens = preprocess_text(ground, join=False)
    prediction_tokens = preprocess_text(prediction, join=False)
    common_tokens = set(ground_tokens) & set(prediction_tokens)
    if len(common_tokens) == 0:
        return 0
    precision = len(common_tokens) / len(prediction_tokens)
    recall = len(common_tokens) / len(ground_tokens)
    return 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0


def calculate_f1_accuracy(grounds: List[str], predictions: List[str]) -> float:
    """
    Calculate the F1 score of the model predictions.
    """
    f1_scores = []
    for m in predictions:
        max_f1 = 0
        for a in grounds:
            if a and m:
                f1 = f1_score(a, m)
                if f1 > max_f1:
                    max_f1 = f1
        f1_scores.append(max_f1)
    return np.mean(f1_scores)