from torch.utils.hipify.hipify_python import preprocessor
from transformers import Trainer, TrainingArguments, Seq2SeqTrainer,AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate

from src.eval_pre_train import compute_metrics

training_args = TrainingArguments(
    output_dir = "./output",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate = 5e-5,
    per_device_train_batch_size = 2,
    per_device_eval_batch_size = 2,
    num_train_epochs = 1,
    weight_decay = 0.01,
    save_total_limit = 2,
    predict_with_generate = True,
    logging_dir="./logs",
    logging_steps=10,
    overwrite_output_dir=True
)

trainer = Seq2SeqTrainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_splits["train"],
    eval_dataset = tokenized_splits["validation"],
    tokenizer = tokenizer,
    compute_metrics = compute_metrics
)

trainer.train()
results = trainer.evaluate()
print(results)
