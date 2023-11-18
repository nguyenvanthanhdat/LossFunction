import json
import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer
from loss_nli.data import data
from sentence_transformers import SentenceTransformer, SentencesDataset, losses
from sentence_transformers.readers import InputExample
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
import numpy as np
import evaluate

dataset = data.ViNLI(tokenizer_name='xlmr', max_length=512).get_dataset()
check_point = "xlm-roberta-large"
model = AutoModelForSequenceClassification.from_pretrained(check_point, num_labels=3)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

device = torch.device("cuda:0")

training_args = TrainingArguments(
    output_dir="model/xlmr/vinli/100",
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    evaluation_strategy="steps",
    logging_dir="logging",
    logging_steps=100,
    num_train_epochs=15,
    report_to="tensorboard",
    learning_rate=1e-4
)
training_args.device

trainer = Trainer(
    model,
    training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset["dev"],
    
    # data_collator=data_collator,
    # tokenizer=tokenizer,
)

trainer.train()