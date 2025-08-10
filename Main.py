from transformers import BartTokenizer
import torch
import re

def clean_scientific_text(text: str) -> str:
    # ​​ Rimuove tag HTML (se il testo viene da file .html)
    text = re.sub(r'<.*?>', ' ', text)

    # ​​ Rimuove hiperlink (URL) e indirizzi email
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)

    # ​​ Rimuove citazioni numeriche tipo [1], [12, 34] (ben formate)
    text = re.sub(r'\[\s*\d+(?:\s*,\s*\d+)*\s*\]', ' ', text)

    # ​​ Rimuove riferimenti autore-anno tipo (Smith et al., 2020)
    text = re.sub(r'\([A-Za-z][A-Za-z\s\.\-]*et al\.,\s*\d{4}\)', ' ', text)

    # ​​ Rimuove formule inline LaTeX come $...$ (da più attento a $$...$$)
    text = re.sub(r'\${1,2}.*?\${1,2}', ' ', text)

    # ​​ Rimuove testo tra parentesi quadre o tonde (generico)
    text = re.sub(r'\[.*?\]|\(.*?\)', ' ', text)

    # ​​ Rimuove simboli, punteggiatura non alfanumerica (ma conserva spazi e lettere accentate)
    text = re.sub(r'[^a-zA-Z0-9À-ž\s]', ' ', text)

    # ​​ Minimizza spazi multipli e strip
    text = re.sub(r'\s+', ' ', text).strip()

    # ​​ Opzionale: lowercase (decidi se serve al tuo caso)
    # text = text.lower()

    return text


# 1️⃣ Carichiamo il tokenizer di BART (pre-addestrato)
# Puoi cambiare 'facebook/bart-base' con 'facebook/bart-large' se vuoi un modello più grande
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

# 2️⃣ Esempio di dati — qui uso frasi dummy, ma tu le sostituirai col tuo dataset
testo_input = [
    "Questo è un esempio di input.",
    "Questa è un'altra frase di esempio."
]

# (Opzionale) target/label — solo se hai un compito tipo traduzione o riassunto
testo_target = [
    "Esempio di output.",
    "Secondo esempio di output."
]

# 3️⃣ Tokenizzazione input
inputs = tokenizer(
    testo_input,
    max_length=64,           # lunghezza massima, da regolare
    padding="max_length",    # padding per uniformare le sequenze
    truncation=True,         # tronca se supera max_length
    return_tensors="pt"      # ritorna tensori PyTorch
)

# 4️⃣ Tokenizzazione target (solo se serve in addestramento)
with tokenizer.as_target_tokenizer():
    labels = tokenizer(
        testo_target,
        max_length=64,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

# 5️⃣ Output pronto
print("Input IDs:\n", inputs["input_ids"])
print("Attention mask:\n", inputs["attention_mask"])
print("Labels:\n", labels["input_ids"])
