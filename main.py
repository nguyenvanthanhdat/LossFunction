import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer
from src.loss_nli.data import data
from src.loss_nli.trainer import trainer
from sentence_transformers import SentenceTransformer, SentencesDataset, losses
from sentence_transformers.readers import InputExample
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
import itertools
import numpy as np
import evaluate
from datasets import load_metric
import os

# def preprocess_fn(sent1, sent2, label):
#     # new_sent = "['CLS'] " + sent2 + " ['SEP'] " + sent1
#     new_sent = sent2 + " </s> " + sent1
#     return new_sent, label

dataset_list = ["ViNLI",
"SNLI",
"MultiNLI",
"Contract_NLI",]

model_dict = {
    'xlmr': "xlm-roberta-large",
    't5': "t5-large",
    'phobert': "vinai/phobert-large",
}

trainer_list = ["CrossEntropyLossTrainer",
                "TripletLossTrainer",
                "ContrastiveLossTrainer",
                "CosineSimilarityLossTrainer"]

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# TrainingArguments
per_device_train_batch_size=int(16),
learning_rate=float(2e-4),
logging_steps=int(10),
eval_steps=int(100),
num_train_epochs=int(1),
weight_decay=float(0.01),
metric_for_best_model = "accuracy"
metric = evaluate.load("accuracy")

for dataset_name, model_name, trainer_name in itertools.product(dataset_list, list(model_dict.keys()),trainer_list):
    print("_______Training dataset", dataset_name, "with model", model_name,"______")
    
    #Init dataset with model tokenizer
    dataset = data.__dict__[dataset_name](tokenizer_name=model_name, num_sample=10).get_dataset()
    dataset = dataset.class_encode_column("labels")

    #Init model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dict[model_name], 
        num_labels= len(np.unique(dataset['train']['labels'])), #ViNLI can have 4 labels
        device_map="auto")

    #Init model's tokenizer and data collator
    tokenizer = AutoTokenizer.from_pretrained(model_dict[model_name])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    training_args = TrainingArguments(
        output_dir=os.path.join("model",model_name, dataset_name, str(num_train_epochs)),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=per_device_train_batch_size[0],
        learning_rate=learning_rate[0],
        evaluation_strategy="steps",
        logging_dir="logging",
        logging_steps=logging_steps[0],
        eval_steps=eval_steps[0],
        num_train_epochs=num_train_epochs[0],
        weight_decay=weight_decay[0],
        report_to="wandb",
        run_name="_".join(str(x) for x in [dataset_name, model_name, num_train_epochs[0], '%.0e' % learning_rate[0]]),
        # disable_tqdm=True,
        metric_for_best_model = metric_for_best_model,
        greater_is_better=True,
        optim= "adamw_torch",
        # label_names=['0','1','2'],
    )

    #Init custom trainer
    trainer = trainer.__dict__[trainer_name](
        model,
        args=training_args,
        train_dataset=dataset['train'],#
        eval_dataset=dataset["dev"],#
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        tokenizer=tokenizer,#
        
    )
    trainer.train()