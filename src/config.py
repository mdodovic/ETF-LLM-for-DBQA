import os
import torch

# Configuration settings
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

PDF_PATH = "../pdfs/Pravilnik_o_OAS_preciscen_jun_2023.pdf"
EMBEDDINGS_FILE = "../embeddings.npy"
SENTENCES_FILE = "../sentences.txt"
METADATA_FILE = "../metadata.json"
HF_TOKEN = "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"