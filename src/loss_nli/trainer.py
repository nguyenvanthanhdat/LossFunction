from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

from torch.nn import CrossEntropyLoss, TripletMarginWithDistanceLoss, functional


class CrossEntropyLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
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

class CosineSimilarityLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = CosineSimilarityLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss