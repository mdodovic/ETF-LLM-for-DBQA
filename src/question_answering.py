from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def find_most_relevant_text(query, sentences, embeddings, model, window_size=2):
    query_embedding = model.encode([query])
    similarities = [cosine_similarity(query_embedding, [emb])[0][0] for emb in embeddings]
    best_match_idx = np.argmax(similarities)
    return " ".join(sentences[max(0, best_match_idx - window_size): min(len(sentences), best_match_idx + window_size + 1)])

def sliding_window_answer(question, context, qa_pipeline, window_size=150):
    words = context.split()
    best_answer, best_score = "", 0
    for i in range(0, len(words), window_size // 2):
        result = qa_pipeline(question=question, context=" ".join(words[i:i + window_size]))
        if result["score"] > best_score:
            best_answer, best_score = result["answer"], result["score"]
    return best_answer