import json
import os
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def clean_text(text):
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\d+\s?", "", text)
    return text.strip()


def save_metadata(pdf_path, metadata_file):
    metadata = {"pdf_last_modified": os.path.getmtime(pdf_path)}
    with open(metadata_file, "w") as f:
        json.dump(metadata, f)


def load_metadata(metadata_file):
    return json.load(open(metadata_file)) if os.path.exists(metadata_file) else {}


def compute_cosine_similarity(pred, truth, model):
    return cosine_similarity(model.encode([pred]), model.encode([truth]))[0][0]


def save_embeddings(sentences, embeddings, sentences_file, embeddings_file):
    with open(sentences_file, "w", encoding="utf-8") as f:
        f.writelines("\n".join(sentences))
    np.save(embeddings_file, embeddings)


def load_embeddings(sentences_file, embeddings_file):
    if os.path.exists(sentences_file) and os.path.exists(embeddings_file):
        with open(sentences_file, "r", encoding="utf-8") as f:
            sentences = f.read().splitlines()
        embeddings = np.load(embeddings_file)
        return sentences, embeddings
    return None, None
