from loss_nli.data import data
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
import os
from datasets import Dataset

data_path = 'data/vinli/UIT_ViNLI_1.0_test.jsonl'
features = ['sentence1', 'sentence2', 'gold_label']
bs = 8
def preprocess_fn(sent1, sent2, label):
    # new_sent = "['CLS'] " + sent2 + " ['SEP'] " + sent1
    new_sent = sent2 + " </s> " + sent1
    return new_sent, label
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")



vinli = data.ViNLI(
    data_path=[data_path],
    preprocess_fn=preprocess_fn,
    features=features,
    tokenizer=tokenizer,
    max_length=256,
    bs=bs)

dataset = vinli.transform_dataset()

dataset = Dataset.from_list(dataset)

id2label = {
    0: "contradiction",
    1: "neutral",
    2: "entailment",
    3: "other"
}

label2id = {
    "contradiction": 0,
    "neutral": 1,
    "entailment": 2,
    "other": 3
}

model = AutoModelForTokenClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=len(id2label), id2label=id2label, label2id=label2id
)

print(os.getcwd())
training_args = TrainingArguments(
    output_dir="model/",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    # push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    # train_dataset=tokenized_wnut["train"],
    train_dataset=dataset,
    # eval_dataset=tokenized_wnut["test"],
    tokenizer=tokenizer,
    # data_collator=data_collator,
    # compute_metrics=compute_metrics,
)

trainer.train()