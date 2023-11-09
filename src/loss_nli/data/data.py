from .base import *
from datasets import load_dataset
from transformers import AutoTokenizer
import os

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


class ViNLI(BaseDataset):
    def __init__(self, tokenizer_name, max_length):
        self.tokenize_name = tokenizer_name
        self.max_length = max_length
        self.data_path = f'data_tokenized/{self.tokenize_name}/vinli/{max_length}'

    def load_from_disk(self):
        data_files = {}
        for split in split_dict:
            path = dataset_path_dict['ViNLI'].format(split=split)
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
        if os.path.isdir(self.data_path):
            dataset = DatasetDict.load_from_disk(self.data_path)
        else:
            self.save_disk()
            dataset = DatasetDict.load_from_disk(self.data_path)
        return dataset['train'], dataset['dev'], dataset['test']
    
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
        if os.path.isdir(self.data_path):
            dataset = DatasetDict.load_from_disk(self.data_path)
        else:
            self.save_disk()
            dataset = DatasetDict.load_from_disk(self.data_path)
        return dataset['train'], dataset['dev'], dataset['test']
    
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
        if os.path.isdir(self.data_path):
            dataset = DatasetDict.load_from_disk(self.data_path)
        else:
            self.save_disk()
            dataset = DatasetDict.load_from_disk(self.data_path)
        return dataset['train'], dataset['dev_matched'], dataset['dev_mismatched']
    
class Contract_NLI(BaseDataset):
    def __init__(self, tokenizer_name, max_length):
        self.tokenize_name = tokenizer_name
        self.max_length = max_length
        self.data_path = f'data_tokenized/{self.tokenize_name}/contract_nli/{max_length}'
    def load_from_disk(self):
        data_files = {}
        for split in split_dict:
            path = dataset_path_dict['Contract_NLI'].format(split=split)
            data_files[split] = path
            print(path)
        dataset = load_dataset("json", data_files=data_files, streaming=True)
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
        if os.path.isdir(self.data_path):
            dataset = DatasetDict.load_from_disk(self.data_path)
        else:
            self.save_disk()
            dataset = DatasetDict.load_from_disk(self.data_path)
        return dataset['train'], dataset['dev'], dataset['test']