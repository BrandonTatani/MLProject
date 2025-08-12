import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.model_selection import train_test_split
from pathlib import Path
import re

# ====== CONFIG ======
DATA_DIR = Path("data")
MODEL_NAME = "facebook/bart-base"
MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 256


# ====== FUNZIONE: Pulizia testo ======
def clean_scientific_text(text):
    if not isinstance(text, str):
        return ""
    # Rimuove spazi multipli
    text = re.sub(r'\s+', ' ', text)
    # Rimuove riferimenti tipo [1], [12], ecc.
    text = re.sub(r'\[\d+\]', '', text)
    # Rimuove caratteri non utili (tranne punteggiatura base)
    text = re.sub(r'[^a-zA-Z0-9.,;:!?\'\"()\s-]', '', text)
    # Elimina spazi iniziali/finali
    return text.strip()


# ====== FUNZIONE: Preprocess per tokenizzazione ======
def preprocess(batch):
    model_inputs = tokenizer(
        batch["article"],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length"
    )
    labels = tokenizer(
        batch["abstract"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


if __name__ == '__main__':
    # 1. Carica modello e tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # 2. Carica dataset concatenato
    df = pd.read_csv(DATA_DIR / 'dataset.csv', header=0, dtype=str)

    # 3. Pulizia testo
    df['article'] = df['article'].apply(clean_scientific_text)
    df['abstract'] = df['abstract'].apply(clean_scientific_text)

    # 4. Conversione in Dataset HuggingFace
    dataset = Dataset.from_pandas(df)

    # 5. Split train (80%) / temp (20%)
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)

    # 6. Split temp in validation (10%) / test (10%)
    temp_split = split_dataset['test'].train_test_split(test_size=0.5, seed=42)

    dataset_splits = {
        'train': split_dataset['train'],
        'validation': temp_split['train'],
        'test': temp_split['test']
    }

    # 7. Salvataggio CSV puliti
    for split_name, split_data in dataset_splits.items():
        split_df = pd.DataFrame(split_data)
        split_df.to_csv(DATA_DIR / f"{split_name}_clean.csv", index=False)
        print(f"✅ Salvato {split_name}_clean.csv con {len(split_df)} righe")

    # 8. Tokenizzazione di tutti gli split
    tokenized_splits = {}
    for split_name, split_data in dataset_splits.items():
        tokenized_splits[split_name] = split_data.map(preprocess, batched=True)

    print("✅ Tokenizzazione completata per train/val/test")
