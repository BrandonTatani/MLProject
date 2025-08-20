from settings import *
from transformers import DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, AutoTokenizer
from preprocessing import Preprocessor
import evaluate

#!!! non so come aggiornare il file requirements ma bisogna installare evaluate

#Inizializzazione
preprocess = Preprocessor()

# Carica modello e tokenizer
print("\rLoading tokenizer...", end='')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("\rLoading model...", end='')
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Utilizzare il DataCollator per generare batch uniformi
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
)

#ROUGE per valutare il modello BART pre-trained
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # Decodifica i token in testo leggibile
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Calcolo ROUGE
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_aggregator = False)

    return result

eval_dataset = preprocess.splits['validation']