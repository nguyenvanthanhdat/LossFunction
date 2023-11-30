# import json
import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer
from loss_nli.data import data
from sentence_transformers import SentenceTransformer, SentencesDataset, losses
from sentence_transformers.readers import InputExample
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import load_metric
import numpy as np
import evaluate
from datasets import load_metric
import os

os.system("wandb login 138c38699b36fb0223ca0f94cde30c6d531895ca")
os.environ["WANDB_PROJECT"] = "Loss-Function"

dataset = data.ViNLI(tokenizer_name='xlmr').get_dataset()
dataset = dataset.class_encode_column("labels")
check_point = "xlm-roberta-large"
model = AutoModelForSequenceClassification.from_pretrained(
    check_point, 
    num_labels=3,
    torch_dtype=torch.float16,
    device_map="auto",)

metric = evaluate.load("accuracy")
# metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# device = torch.device("cuda:0,1")
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="model/xlmr/vinli/10",
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=16,
    learning_rate=2e-4,
    evaluation_strategy="steps",
    logging_dir="logging",
    logging_steps=10,
    eval_steps=100,
    num_train_epochs=10,
    weight_decay=0.01,
    report_to="wandb",
    run_name="xlmr_vinli_50_2e-7",
    # disable_tqdm=True,
    metric_for_best_model = "accuracy",
    greater_is_better=True,
    optim= "adamw_torch",
    # label_names=['0','1','2'],
    
)
training_args

trainer = Trainer(
    model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset["dev"],
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()