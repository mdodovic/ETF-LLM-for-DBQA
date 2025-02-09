import os
import re
import json

import torchvision
torchvision.disable_beta_transforms_warning()
import transformers
from huggingface_hub import login
from langchain_community.document_loaders import PDFMinerLoader
from pdf2image import convert_from_path
import pytesseract
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

transformers.logging.set_verbosity_error()

# Set device to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# Hugging Face Login
hf_token = "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"
login(token=hf_token)
print("Successfully logged into Hugging Face!")

# File paths for storage
pdf_path = "pdfs/Pravilnik_o_OAS_preciscen_jun_2023.pdf"  # Ensure the path is correct
embeddings_file = "embeddings.npy"
sentences_file = "sentences.txt"
metadata_file = "metadata.json"

# Load pre-trained embedding model (use CUDA if available)
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device=device)


def compute_cosine_similarity(pred, truth):
    """Computes cosine similarity between the predicted and correct answer embeddings."""
    pred_embedding = model.encode([pred])
    truth_embedding = model.encode([truth])

    similarity = cosine_similarity(pred_embedding, truth_embedding)[0][0]  # Get similarity score
    return similarity


def clean_text(text):
    """Cleans extracted OCR text by removing artifacts like page numbers and unnecessary line breaks."""
    text = re.sub(r"\n+", " ", text)  # Replace multiple newlines with a space
    text = re.sub(r"\s+", " ", text)  # Normalize spaces
    text = re.sub(r"\d+\s?", "", text)  # Remove page numbers and artifacts
    return text.strip()


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF using PDFMinerLoader or fallback to OCR."""
    try:
        loader = PDFMinerLoader(pdf_path)
        documents = loader.load()
        text = "\n".join(doc.page_content for doc in documents)
        if text.strip():
            return clean_text(text)  # Return if PDFMiner extracted text successfully
    except Exception as e:
        print(f"Document is scanned: {e}")
        print("Using OCR for scanned PDF...")
        return clean_text(extract_text_with_ocr(pdf_path))
    return None


def extract_text_with_ocr(pdf_path):
    """Extract text from a scanned PDF using Tesseract OCR."""
    text = ""
    images = convert_from_path(pdf_path)  # Convert PDF pages to images
    for img in images:
        text += pytesseract.image_to_string(img) + "\n"
    return clean_text(text)


def save_metadata(pdf_path, metadata_file):
    """Save the metadata of the PDF (e.g., last modified time)."""
    metadata = {"pdf_last_modified": os.path.getmtime(pdf_path)}
    with open(metadata_file, "w") as f:
        json.dump(metadata, f)


def load_metadata(metadata_file):
    """Load stored metadata if available."""
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            return json.load(f)
    return {}


def save_embeddings(sentences, embeddings, sentences_file, embeddings_file):
    """Save sentences and their embeddings to files."""
    with open(sentences_file, "w", encoding="utf-8") as f:
        f.writelines("\n".join(sentences))
    np.save(embeddings_file, embeddings)


def load_embeddings(sentences_file, embeddings_file):
    """Load sentences and their embeddings from files if they exist."""
    if os.path.exists(sentences_file) and os.path.exists(embeddings_file):
        with open(sentences_file, "r", encoding="utf-8") as f:
            sentences = f.read().splitlines()
        embeddings = np.load(embeddings_file)
        return sentences, embeddings
    return None, None


# Load stored metadata
metadata = load_metadata(metadata_file)

# Check if embeddings exist and PDF has not changed
if os.path.exists(embeddings_file) and metadata.get("pdf_last_modified") == os.path.getmtime(pdf_path):
    print("Loading precomputed embeddings...")
    sentences, embeddings = load_embeddings(sentences_file, embeddings_file)
else:
    print("Extracting text and computing embeddings...")
    pdf_text = extract_text_from_pdf(pdf_path)
    sentences = pdf_text.split(".")  # Split into sentences
    embeddings = model.encode(sentences)

    # Save results for future use
    save_embeddings(sentences, embeddings, sentences_file, embeddings_file)
    save_metadata(pdf_path, metadata_file)


def find_most_relevant_text(query, sentences, embeddings, model, window_size=2):
    """Find the most relevant text snippet and extend its context."""
    query_embedding = model.encode([query])

    # Compute cosine similarity for each sentence
    similarities = [cosine_similarity(query_embedding, np.array([emb]))[0][0] for emb in embeddings]
    best_match_idx = np.argmax(similarities)

    # Extend context by including sentences before and after the best match
    start_idx = max(0, best_match_idx - window_size)
    end_idx = min(len(sentences), best_match_idx + window_size + 1)

    extended_context = " ".join(sentences[start_idx:end_idx])
    return extended_context  # Return extended snippet


# def answer_question(question, context):
#     """Generate an answer using the QA model."""
#     result = qa_pipeline(question=question, context=context)
#     return result["answer"]
# # answer = answer_question(query, relevant_context)
# print("Answer:", answer)

def sliding_window_answer(question, context, qa_pipeline, window_size=150):
    """Split the context into smaller windows to improve QA model accuracy."""
    words = context.split()
    best_answer = ""
    best_score = 0

    for i in range(0, len(words), window_size // 2):  # Overlapping windows
        sub_context = " ".join(words[i:i + window_size])
        result = qa_pipeline(question=question, context=sub_context)

        if result["score"] > best_score:  # Keep the best answer
            best_score = result["score"]
            best_answer = result["answer"]

    return best_answer


questions = [
    "Колико има студијских програма на основним академским студијама?",
    "Која су два акредитована студијска програма на основним академским студијама?",
    "Колико радних недеља има школска година?",
    "Шта су консултације?",
    "Када седница има кворум?",
    "Шта су основни подаци о предмету?",
    "Шта је циљ семинара?",
    "Шта је циљ пројекта?"
]

ground_truth_answers = [
    "Два",
    "Електротехника и рачунарство и Софтверско инжењерство",
    "Школска година има 42 радне недеље",
    "Консултације су облик индивидуалне наставе којим наставник или сарадник, у непосредном контакту са студентом, појашњава студенту сложене делове градива и пружа помоћ у решавању одређених проблема, као и при изради семинарских и завршних радова",
    "Седница има кворум уколико јој присуствује више од половине укупног броја чланова Комисије",
    "Назив, година, фонд часова, број ЕСПБ бодова",
    "Циљ семинара је да се дубље и критички размотре неки тематски садржаји",
    "Циљ пројеката је увођење студената у инжењерски рад."
]

predicted_answers = []

for query in questions:
    # query = input(" Enter question: " )
    # if query=="exit":
    #     break

    # Query example

    relevant_context = find_most_relevant_text(query, sentences, embeddings, model, window_size=2)

    print("Extended Context:", relevant_context)

    # Load QA model (use GPU if available)
    qa_pipeline = pipeline("question-answering", model="deepset/xlm-roberta-large-squad2", device=0 if device == "cuda" else -1)

    # Generate answer
    print("Question:", query)

    answer = sliding_window_answer(query, relevant_context, qa_pipeline)
    print("Answer:", answer)

    predicted_answers.append(answer)
    print()

# Compute cosine similarity for each pair
cosines = []
for i, (pred, truth) in enumerate(zip(predicted_answers, ground_truth_answers)):
    cosine_sim = compute_cosine_similarity(pred, truth)
    cosines.append(cosine_sim)
    print(f"QA Pair {i+1}: Cosine Similarity = {cosine_sim:.4f}")

print("Avg:", np.mean(cosines))
print("Med:", np.median(cosines))

