from .base import *
from datasets import load_dataset
from transformers import AutoTokenizer
import os
import pandas as pd
from datasets import concatenate_datasets
dataset_path_dict = {
    'ViNLI': 'data/vinli/UIT_ViNLI_1.0_{split}.jsonl',
    'SNLI': 'data/snli/snli_1.0_{split}.jsonl',
    'MultiNLI': 'data/multinli/multinli_1.0_{split}.jsonl',
    'Contract_NLI':'data/contract-nli/{split}.json'
}   
split_dict = ['train', 'test', 'dev']
tokenizer_dict = {
    'xlmr': "xlm-roberta-large",
    't5': "t5-large",
    'phobert': "vinai/phobert-large",
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

def Find_max_length(dataset, split_dict, tokenize_name):
    # tokenizer = AutoTokenizer.from_pretrained(tokenizer_dict[tokenize_name])
    tokenizer = AutoTokenizer.from_pretrained(tokenize_name)
    dataset['train'] = dataset['train'].map(lambda examples: tokenizer(
        examples["sentence2"], 
        examples["sentence1"],
        ), batched=True)
    # Mergedata = concatenate_datasets([dataset[split_dict[0]],dataset[split_dict[1]],dataset[split_dict[2]]])
    # sorted_sequences = sorted(enumerate(Mergedata['attention_mask']), key=lambda x: len(x[1]), reverse=True)
    sorted_sequences = sorted(enumerate(dataset['train']['attention_mask']), key=lambda x: len(x[1]), reverse=True)
    sorted_indices, sorted_sequences = zip(*sorted_sequences)
    return len(sorted_sequences[0])

def TakeSampleDataset(dataset, split_dict, num_sample):
    dataset[split_dict[0]] = dataset[split_dict[0]].select(range(num_sample))
    dataset[split_dict[1]] = dataset[split_dict[0]].select(range(num_sample))
    dataset[split_dict[2]] = dataset[split_dict[0]].select(range(num_sample))
    return dataset

class ViNLI(BaseDataset):
    def __init__(self, tokenizer_name, load_all_labels=False, num_sample = 0):
        self.tokenize_name = tokenizer_name
        self.load_all_labels = load_all_labels
        self.max_length = None
        self.data_path = None
        self.num_sample = num_sample # this will determine to load all data or just k sample data

    def load_from_disk(self):
        data_files = {}
        for split in split_dict:
            path = dataset_path_dict['ViNLI'].format(split=split)
            data_files[split] = path
        dataset = load_dataset("json", data_files=data_files).filter(lambda example: example['gold_label'] != '-')
        if not self.load_all_labels:
            dataset = dataset.filter(lambda example: example['gold_label'] != 'other')
        dataset = dataset.map(lambda example: {"labels": label_dict[example["gold_label"]]}, remove_columns=["gold_label"])
        self.max_length = Find_max_length(dataset, split_dict, self.tokenize_name)
        self.data_path = f'data_tokenized/{self.tokenize_name}/vinli/{self.max_length}'
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
            return_tensors="pt",
            ), batched=True)
        return dataset
    
    def save_disk(self):
        dataset = self.tokenize()
        if not os.path.isdir(self.data_path):
            dataset.save_to_disk(self.data_path)
    
    def get_dataset(self) -> Tuple[Dataset]:
        self.save_disk()
        dataset = DatasetDict.load_from_disk(self.data_path)
        dataset = dataset.remove_columns(['pairID', 'link', 'context', 'sentence1', 'sentenceID', 'topic', 'sentence2', 'annotator_labels'])
        if self.num_sample!=0 :
            dataset = TakeSampleDataset(dataset, split_dict, self.num_sample)
        return dataset
    
class SNLI(BaseDataset):
    def __init__(self, tokenizer_name, load_all_labels = True, num_sample = 0):
        self.tokenize_name = tokenizer_name
        self.max_length = None
        self.data_path = None
        self.num_sample = num_sample # this will determine to load all data or just k sample data

    def load_from_disk(self):
        data_files = {}
        for split in split_dict:
            path = dataset_path_dict['SNLI'].format(split=split)
            data_files[split] = path
        dataset = load_dataset("json", data_files=data_files)
        dataset = dataset.map(lambda example: {"labels": label_dict[example["gold_label"]]}, remove_columns=["gold_label"])
        self.max_length = Find_max_length(dataset, split_dict, self.tokenize_name)
        self.data_path = f'data_tokenized/{self.tokenize_name}/snli/{self.max_length}'
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
            return_tensors="pt",
            ), batched=True)
        return dataset
    
    def save_disk(self):
        dataset = self.tokenize()
        if not os.path.isdir(self.data_path):
            dataset.save_to_disk(self.data_path)

    def get_dataset(self) -> Tuple[Dataset]:
        self.save_disk()
        dataset = DatasetDict.load_from_disk(self.data_path)
        dataset = dataset.remove_columns(['captionID', 'pairID','sentence1', 'sentence1_binary_parse', 'sentence1_parse', 'sentence2', 'sentence2_binary_parse', 'sentence2_parse', 'annotator_labels'])
        if self.num_sample!=0 :
            dataset = TakeSampleDataset(dataset, split_dict, self.num_sample)
        return dataset
    
class MultiNLI(BaseDataset):
    def __init__(self, tokenizer_name, load_all_labels = True, num_sample=0):
        self.tokenize_name = tokenizer_name
        self.max_length = None
        self.data_path = None
        self.split_dict =['dev_matched','dev_mismatched','train']
        self.num_sample = num_sample # this will determine to load all data or just k sample data

    def load_from_disk(self):
        data_files = {}
        for split in self.split_dict:
            # print(split)
            path = dataset_path_dict['MultiNLI'].format(split=split)
            data_files[split] = path
        dataset = load_dataset("json", data_files=data_files)
        dataset = dataset.map(lambda example: {"labels": label_dict[example["gold_label"]]}, remove_columns=["gold_label"])
        self.max_length = Find_max_length(dataset, split_dict, self.tokenize_name)
        self.data_path = f'data_tokenized/{self.tokenize_name}/multinli/{self.max_length}'
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
        if not os.path.isdir(self.data_path):
            dataset.save_to_disk(self.data_path)
            
    def get_dataset(self) -> Tuple[Dataset]:
        self.save_disk()
        dataset = DatasetDict.load_from_disk(self.data_path)
        dataset = dataset.remove_columns(['promptID', 'pairID','sentence1', 'sentence1_binary_parse', 'sentence1_parse', 'sentence2', 'sentence2_binary_parse', 'sentence2_parse', 'annotator_labels','genre'])
        if self.num_sample!=0 :
            dataset = TakeSampleDataset(dataset, self.split_dict, self.num_sample)
        return dataset
    
class Contract_NLI(BaseDataset):
    def __init__(self, tokenizer_name, load_all_labels = True, num_sample = 0):
        self.tokenize_name = tokenizer_name
        self.max_length = None
        self.data_path = None
        self.num_sample = num_sample # this will determine to load all data or just k sample data

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
            df = pd.DataFrame(rows, columns=['doc_id', 'sentence1', 'sentence2', 'label', 'spans'])
            return df
        dataset = DatasetDict()
        for split in split_dict:
            path = dataset_path_dict['Contract_NLI'].format(split=split)
            dataset[split]= Dataset.from_dict(parse_file(path))
        dataset= dataset.map(lambda example: {"labels": contract_label_dict[example["label"]]}, remove_columns=["label"])
        self.max_length = Find_max_length(dataset, split_dict, self.tokenize_name)
        self.data_path = f'data_tokenized/{self.tokenize_name}/contractnli/{self.max_length}'
        return dataset
    
    def tokenize(self) -> Tuple[Dataset]:
        dataset = self.load_from_disk()
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dict[self.tokenize_name])
        dataset = dataset.map(lambda examples: 
            tokenizer(
            examples["sentence2"], 
            examples["sentence1"],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"), batched=True)
        return dataset
    
    def save_disk(self):
        dataset = self.tokenize()
        if not os.path.isdir(self.data_path):
            dataset.save_to_disk(self.data_path)
            
    def get_dataset(self) -> Tuple[Dataset]:
        self.save_disk()
        dataset = DatasetDict.load_from_disk(self.data_path)
        dataset = dataset.remove_columns(['doc_id','sentence1', 'sentence2', 'spans'])
        if self.num_sample!=0 :
            dataset = TakeSampleDataset(dataset, split_dict, self.num_sample)
        return dataset