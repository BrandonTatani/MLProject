from settings import *
from transformers import DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from preprocessing import Preprocessor, tokenizer, model
import evaluate
from tqdm import tqdm

#pip install absl-py nltk rouge-score
#pip install "accelerate>=0.26.0"
#pip install --upgrade transformers datasets evaluate accelerate
#pip install --upgrade "transformers>=4.39.0" "accelerate>=0.26.0" datasets evaluate

#pip freeze > requirements.txt

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

    result = {k: round(v * 100, 2) for k, v in result.items()}

    return result

if __name__ == "__main__":
    eval_dataset = preprocess.splits['validation']

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    do_train=False,
    do_eval=True,
    logging_dir="./logs",
)

# Inizializza il trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    eval_dataset=eval_dataset,
)

# Valutazione
metrics = trainer.evaluate()
print("📊 Risultati ROUGE (con Seq2SeqTrainer):", metrics)