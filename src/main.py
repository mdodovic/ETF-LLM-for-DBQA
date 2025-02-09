import transformers

from sentence_transformers import SentenceTransformer
from transformers import pipeline
from huggingface_hub import login
from config import *
from utils import *
from text_extraction import extract_text_from_pdf
from question_answering import find_most_relevant_text, sliding_window_answer

transformers.logging.set_verbosity_error()

login(token=HF_TOKEN)
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device=device)
qa_pipeline = pipeline("question-answering", model="deepset/xlm-roberta-large-squad2", device=device)
metadata = load_metadata(METADATA_FILE)

if os.path.exists(EMBEDDINGS_FILE) and metadata.get("pdf_last_modified") == os.path.getmtime(PDF_PATH):
    sentences, embeddings = load_embeddings(SENTENCES_FILE, EMBEDDINGS_FILE)
else:
    pdf_text = extract_text_from_pdf(PDF_PATH)
    sentences = pdf_text.split(".")
    embeddings = model.encode(sentences)
    save_embeddings(sentences, embeddings, SENTENCES_FILE, EMBEDDINGS_FILE)
    save_metadata(PDF_PATH, METADATA_FILE)


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

qa_pipeline = pipeline("question-answering", model="deepset/xlm-roberta-large-squad2",
                       device=0 if device == "cuda" else -1)

for query in questions:
    relevant_context = find_most_relevant_text(query, sentences, embeddings, model, window_size=2)

    print("Extended Context:", relevant_context)

    # Generate answer
    print("Question:", query)

    answer = sliding_window_answer(query, relevant_context, qa_pipeline)
    print("Answer:", answer)

    predicted_answers.append(answer)
    print()

# Compute cosine similarity for each pair
cosines = []
for i, (pred, truth) in enumerate(zip(predicted_answers, ground_truth_answers)):
    cosine_sim = compute_cosine_similarity(pred, truth,model)
    cosines.append(cosine_sim)
    print(f"QA Pair {i+1}: Cosine Similarity = {cosine_sim:.4f}")

print("Avg:", np.mean(cosines))
print("Med:", np.median(cosines))

