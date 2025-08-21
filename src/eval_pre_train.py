from settings import *
from transformers import DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, AutoTokenizer
from preprocessing import Preprocessor, tokenizer, model
import evaluate

#!!! non so come aggiornare il file requirements ma bisogna installare evaluate

#Inizializzazione
preprocess = Preprocessor()

# Utilizzare il DataCollator per generare batch uniformi
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,
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

if __name__ == "__main__":
    eval_dataset = preprocess.splits['validation']