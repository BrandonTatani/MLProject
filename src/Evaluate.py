from transformers import Trainer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from preprocessing import Preprocessor, tokenizer, model
from src.eval_pre_train import compute_metrics
from settings import BASE_DIR
import torch

torch.set_num_threads(8) # using multiple threads based on CPU

preprocessed = Preprocessor()

collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,
)

training_args = Seq2SeqTrainingArguments(
    output_dir = BASE_DIR / "output",
    save_strategy = "epoch",
    eval_strategy="steps",
    eval_steps=500,
    learning_rate = 5e-5,
    per_device_train_batch_size = 4,
    per_device_eval_batch_size = 4,
    num_train_epochs = 1, # maximum number of epochs
    weight_decay = 0.01,
    save_total_limit = 2,
    predict_with_generate = True,
    logging_dir= BASE_DIR / "logs",
    overwrite_output_dir=True,
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model = model,
    args = training_args,
    train_dataset = preprocessed.train(),
    eval_dataset = preprocessed.eval(),
    data_collator = collator,
    tokenizer = tokenizer,
    compute_metrics = compute_metrics
)

trainer.train()
results = trainer.evaluate(preprocessed.test())
print(results)
