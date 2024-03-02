from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
import evaluate
import numpy as np
import torch

from torch.nn import CrossEntropyLoss, CosineSimilarity, TripletMarginWithDistanceLoss, functional

def compute_metrics(eval_pred):
    metric1 = evaluate.load("precision")
    metric2 = evaluate.load("recall")
    metric3 = evaluate.load("f1")
    metric4 = evaluate.load("accuracy")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision = metric1.compute(predictions=predictions, references=labels,
                                average="macro")["precision"]
    recall = metric2.compute(predictions=predictions, references=labels,
                             average="macro")["recall"]
    f1 = metric3.compute(predictions=predictions, references=labels,
                         average="macro")["f1"]
    accuracy = metric4.compute(predictions=predictions, references=labels)[
        "accuracy"]

    return {"precision": precision, "recall": recall, "f1": f1,
            "accuracy": accuracy}

class CrossEntropyLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = CrossEntropyLoss()
        print(logits.view(-1, self.model.config.num_labels).shape)
        print(labels.view(-1).shape)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    
class CosineSimilarityLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = CosineSimilarity()
        num_label = logits.shape[1]
        new_labels = torch.nn.functional.one_hot(labels, num_classes=num_label)
        # print(logits.view(-1, self.model.config.num_labels).shape)
        # print(labels.view(-1).shape)
        # loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        loss = loss_fct(logits, new_labels)
        loss = torch.mean(loss)
        return (loss, outputs) if return_outputs else loss

class TripletLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        anchor_data, positive_data, negative_data = inputs.values()
        anchor_output = self.use_avg_2(anchor_data, model)
        positive_output = self.use_avg_2(positive_data, model)
        negative_output = self.use_avg_2(negative_data, model)
        triplet_loss = TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - functional.cosine_similarity(x, y),margin=0.5)
        # compute custom loss
        loss = triplet_loss(anchor_output, positive_output, negative_output)
        return (loss, outputs) if return_outputs else loss
    
class ContrastiveLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = ContrastiveLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

