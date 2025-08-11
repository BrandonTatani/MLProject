import pandas as pd
import re
from settings import DATA_DIR
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset

def concat_dataset():
    """Dataset was already split and kaggle load was not working"""

    df_1 = pd.read_csv(DATA_DIR / 'train.csv')
    df_2 = pd.read_csv(DATA_DIR / 'validation.csv')
    df_3 = pd.read_csv(DATA_DIR / 'test.csv')

    _df = pd.concat([df_1, df_2, df_3])
    _df.to_csv(DATA_DIR / 'dataset.csv', index=False)

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

    return str(text)

def preprocess(batch):
    articles = [clean_scientific_text(foo) for foo in batch['article']]
    abstracts = [clean_scientific_text(foo) for foo in batch['abstract']]
    inputs = tokenizer(
        articles, # text input is expected to be list[str]
        max_length=1024, # we have to se the length of the articles
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            abstracts,
            max_length=256,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

    inputs['labels'] = labels['input_ids'] # tokenized abstract IDs
    return inputs

if __name__ == '__main__':
    model_name = "facebook/bart-base"
    # model_name = "facebook/bart-large"
    print(f"Selected model: {model_name}")
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Tokenizer...")
    print("Loading Model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print("Model...")

    print("Loading Data...")
    df = pd.read_csv(DATA_DIR / 'dataset.csv', header=0, dtype=str, nrows=5)

    dataset = Dataset.from_pandas(df) # Tokenizer is meant to work with Dataset Object
    processed = dataset.map(preprocess, batched=True)
    print(processed[0])