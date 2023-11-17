from .base import *
from datasets import load_dataset
from transformers import AutoTokenizer
import os
import pandas as pd
dataset_path_dict = {
    'ViNLI': 'data/vinli/UIT_ViNLI_1.0_{split}.jsonl',
    'SNLI': 'data/snli/snli_1.0_{split}.jsonl',
    'MultiNLI': 'data/multinli/multinli_1.0_{split}.jsonl',
    'Contract_NLI':'data/contract-nli/{split}.json'
}   
split_dict = ['train', 'test', 'dev']
tokenizer_dict = {
    'xlmr': "xlm-roberta-large"
}
label_dict = {
    "contradiction": 0,
    "neutral": 1,
    "entailment": 2,
    "other": 3,
    "-": -1
}
contract_label_dict = {
    "Contradiction": 0,
    "NotMentioned": 1,
    "Entailment":2,
}

class ViNLI(BaseDataset):
    def __init__(self, tokenizer_name, max_length, load_all_labels=False):
        self.tokenize_name = tokenizer_name
        self.max_length = max_length
        self.data_path = f'data_tokenized/{self.tokenize_name}/vinli/{max_length}'
        self.load_all_labels = load_all_labels

    def load_from_disk(self):
        data_files = {}
        for split in split_dict:
            path = dataset_path_dict['ViNLI'].format(split=split)
            data_files[split] = path
        dataset = load_dataset("json", data_files=data_files).filter(lambda example: example['gold_label'] != '-')
        if not self.load_all_labels:
            dataset = dataset.filter(lambda example: example['gold_label'] != 'other')
        return dataset

    def tokenize(self) -> Tuple[Dataset]:
        dataset = self.load_from_disk()
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dict[self.tokenize_name])
        dataset = dataset.map(lambda examples: tokenizer(
            examples["sentence2"], 
            examples["sentence1"],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"), batched=True)
        return dataset
    
    def save_disk(self):
        dataset = self.tokenize()
        dataset.save_to_disk(self.data_path)
    
    def get_dataset(self) -> Tuple[Dataset]:
        if not os.path.isdir(self.data_path):
            self.save_disk()
        dataset = DatasetDict.load_from_disk(self.data_path)
        dataset = dataset.map(lambda example: {"labels": label_dict[example["gold_label"]]}, remove_columns=["gold_label"])
        dataset = dataset.remove_columns(['pairID', 'link', 'context', 'sentence1', 'sentenceID', 'topic', 'sentence2', 'annotator_labels'])
        return dataset
    
class SNLI(BaseDataset):
    def __init__(self, tokenizer_name, max_length):
        self.tokenize_name = tokenizer_name
        self.max_length = max_length
        self.data_path = f'data_tokenized/{self.tokenize_name}/snli/{max_length}'
    def load_from_disk(self):
        data_files = {}
        for split in split_dict:
            path = dataset_path_dict['SNLI'].format(split=split)
            data_files[split] = path
        dataset = load_dataset("json", data_files=data_files)
        return dataset
    def tokenize(self) -> Tuple[Dataset]:
        dataset = self.load_from_disk()
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dict[self.tokenize_name])
        dataset = dataset.map(lambda examples: tokenizer(
            examples["sentence2"], 
            examples["sentence1"],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"), batched=True)
        return dataset
    def save_disk(self):
        dataset = self.tokenize()
        dataset.save_to_disk(self.data_path)
    def get_dataset(self) -> Tuple[Dataset]:
        if not os.path.isdir(self.data_path):
            self.save_disk()
        dataset = DatasetDict.load_from_disk(self.data_path)
        dataset = dataset.map(lambda example: {"labels": label_dict[example["gold_label"]]}, remove_columns=["gold_label"])
        dataset = dataset.remove_columns(['captionID', 'pairID','sentence1', 'sentence1_binary_parse', 'sentence1_parse', 'sentence2', 'sentence2_binary_parse', 'sentence2_parse', 'annotator_labels'])
        return dataset
    
class MultiNLI(BaseDataset):
    def __init__(self, tokenizer_name, max_length):
        self.tokenize_name = tokenizer_name
        self.max_length = max_length
        self.data_path = f'data_tokenized/{self.tokenize_name}/multinli/{max_length}'
    def load_from_disk(self):
        data_files = {}
        split_dict =['dev_matched','dev_mismatched','train']
        for split in split_dict:
            # print(split)
            path = dataset_path_dict['MultiNLI'].format(split=split)
            data_files[split] = path
        dataset = load_dataset("json", data_files=data_files)
        return dataset
    def tokenize(self) -> Tuple[Dataset]:
        dataset = self.load_from_disk()
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dict[self.tokenize_name])
        dataset = dataset.map(lambda examples: tokenizer(
            examples["sentence2"], 
            examples["sentence1"],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"), batched=True)
        return dataset
    def save_disk(self):
        dataset = self.tokenize()
        dataset.save_to_disk(self.data_path)
    def get_dataset(self) -> Tuple[Dataset]:
        if not os.path.isdir(self.data_path):
            self.save_disk()
        dataset = DatasetDict.load_from_disk(self.data_path)
        dataset = dataset.map(lambda example: {"labels": label_dict[example["gold_label"]]}, remove_columns=["gold_label"])
        dataset = dataset.remove_columns(['promptID', 'pairID','sentence1', 'sentence1_binary_parse', 'sentence1_parse', 'sentence2', 'sentence2_binary_parse', 'sentence2_parse', 'annotator_labels','genre'])
        return dataset
    
class Contract_NLI(BaseDataset):
    def __init__(self, tokenizer_name, max_length):
        self.tokenize_name = tokenizer_name
        self.max_length = max_length
        self.data_path = f'data_tokenized/{self.tokenize_name}/contract_nli/{max_length}'
    def load_from_disk(self):
        def parse_file(file_name):
            with open(file_name) as f:
                data = json.load(f)
            documents = data['documents']
            labels = data['labels']
            rows = []
            for doc in documents:
                doc_id = doc['id']
                text = doc['text']
                for annotation_key in doc['annotation_sets'][0]['annotations']:
                    hyp_key = annotation_key
                    label = doc['annotation_sets'][0]['annotations'][hyp_key]['choice']
                    spans = doc['annotation_sets'][0]['annotations'][hyp_key]['spans']
                    hyp = labels[hyp_key]['hypothesis']
                    rows.append([doc_id, text, hyp, label, spans])
            df = pd.DataFrame(rows, columns=['doc_id', 'premise', 'hypothesis', 'label', 'spans'])
            return df
        datasetdict = DatasetDict()
        for split in split_dict:
            path = dataset_path_dict['Contract_NLI'].format(split=split)
            datasetdict[split]= Dataset.from_dict(parse_file(path))
        return datasetdict
    def tokenize(self) -> Tuple[Dataset]:
        dataset = self.load_from_disk()
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dict[self.tokenize_name])
        dataset = dataset.map(lambda examples: 
            tokenizer(
            examples["premise"], 
            examples["hypothesis"],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"), batched=True)
        return dataset
    def save_disk(self):
        dataset = self.tokenize()
        dataset.save_to_disk(self.data_path)
    def get_dataset(self) -> Tuple[Dataset]:
        if not os.path.isdir(self.data_path):
            self.save_disk()
        dataset = DatasetDict.load_from_disk(self.data_path)
        dataset = dataset.map(lambda example: {"labels": contract_label_dict[example["label"]]}, remove_columns=["label"])
        dataset = dataset.remove_columns(['doc_id','premise', 'hypothesis', 'spans'])
        return dataset