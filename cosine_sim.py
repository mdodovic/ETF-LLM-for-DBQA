from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the embedding model
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')


def compute_cosine_similarity(pred, truth):
    """Computes cosine similarity between the predicted and correct answer embeddings."""
    pred_embedding = model.encode([pred])
    truth_embedding = model.encode([truth])

    similarity = cosine_similarity(pred_embedding, truth_embedding)[0][0]  # Get similarity score
    return similarity



ground_truth_answers = [
    "најмање 30, а највише 70 поена",
    "Испит мора носити 30 поена",
    "Студент мора имати 51 поен за пролаз",
]

predicted_answers = [
    "70 поена",
    "Испит носи 30 поена",
    "За пролаз треба 51 поен"
]

# Compute cosine similarity for each pair
cosines = []
for i, (pred, truth) in enumerate(zip(predicted_answers, ground_truth_answers)):
    cosine_sim = compute_cosine_similarity(pred, truth)
    cosines.append(cosine_sim)
    # print(f"QA Pair {i+1}: Cosine Similarity = {cosine_sim:.4f}")

print("Avg:", np.mean(cosines))
print("Med:", np.median(cosines))
