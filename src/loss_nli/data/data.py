from .base import *
from datasets import load_dataset
from transformers import AutoTokenizer
import os

dataset_path_dict = {
    'ViNLI': 'data/vinli/UIT_ViNLI_1.0_{split}.jsonl'
}
split_dict = ['train', 'test', 'dev']
tokenizer_dict = {
    'xlmr': "xlm-roberta-large"
}


class ViNLI(BaseDataset):
    def __init__(self, tokenizer_name):
        self.tokenize_name = tokenizer_name
        self.data_path = f'data_tokenized/{self.tokenize_name}/vinli'

    def load_from_disk(self):
        data_files = {}
        for split in split_dict:
            # print(split)
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
            max_length=100,
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

# class ViNLI(BaseDataset):
#     def __init__(self, data_path, preprocess_fn, tokenizer, max_length, bs = 4, features=None):
#         if not isinstance(data_path, list):
#             raise TypeError("Change <data_path> to list. If has 1 path please format \"[<data_path>]\"")
#         self.data_path = data_path
#         self.features = features
#         self.preprocess_fn = preprocess_fn
#         self.tokenizer = tokenizer
#         self.bs = bs
#         self.max_length = max_length

#     def load_dataset(self) -> DataLoader:
#         dataset = DatasetNLI(self.data_path, None)
#         dataloader = DataLoader(dataset, batch_size=1)
#         return dataloader

#     def collapse_dataset(self) -> DataLoader:
#         if self.features == None:
#             raise Exception("<Features not None>")
#         dataset = DatasetNLI(self.data_path, self.features)
#         dataloader = DataLoader(dataset, batch_size=1)
#         return dataloader
    
#     def preprocess(self) -> DataLoader:
#         dataset = PreprocessDataset(self.collapse_dataset(), self.preprocess_fn)
#         dataloader = DataLoader(dataset, batch_size=64)
#         return dataloader

#     def transform_dataset(self) -> DataLoader:
#         dataset = PreprocessDataset(self.collapse_dataset(), self.preprocess_fn)
#         dataloader = DataLoader(dataset, batch_size=1)
#         new_dataset = TokenizeDataset(dataloader, self.tokenizer, max_length=self.max_length)
#         # new_dataloader = DataLoader(new_dataset, batch_size=self.bs)
#         # return new_dataloader
#         print(type(new_dataset))
#         return new_dataset
    
# class SNLI(BaseDataset):
#     def __init__(self, data_path, features=None):
#         if not isinstance(data_path, list):
#             raise TypeError("Change <data_path> to list. If has 1 path please format \"[<data_path>]\"")
#         self.data_path = data_path
#         self.features = features

#     def load_dataset(self) -> DataLoader:
#         dataset = DatasetNLI(self.data_path, None)
#         dataloader = DataLoader(dataset, batch_size=1)
#         return dataloader

#     def collapse_dataset(self) -> DataLoader:
#         if self.features == None:
#             raise Exception("<Features not None>")
#         dataset = DatasetNLI(self.data_path, self.features)
#         dataloader = DataLoader(dataset, batch_size=1)
#         return dataloader
    
#     def preprocess(self) -> DataLoader:
#         pass

# class MultiNLI(BaseDataset):
#     def __init__(self, data_path, features=None):
#         if not isinstance(data_path, list):
#             raise TypeError("Change <data_path> to list. If has 1 path please format \"[<data_path>]\"")
#         self.data_path = data_path
#         self.features = features

#     def load_dataset(self) -> DataLoader:
#         dataset = DatasetNLI(self.data_path, None)
#         dataloader = DataLoader(dataset, batch_size=1)
#         return dataloader

#     def collapse_dataset(self) -> DataLoader:
#         if self.features == None:
#             raise Exception("<Features not None>")
#         dataset = DatasetNLI(self.data_path, self.features)
#         dataloader = DataLoader(dataset, batch_size=1)
#         return dataloader
    
#     def preprocess(self) -> DataLoader:
#         pass

# class ContractNLI(BaseDataset):
#     def __init__(self, data_path, features=None):
#         if not isinstance(data_path, list):
#             raise TypeError("Change <data_path> to list. If has 1 path please format \"[<data_path>]\"")
#         self.data_path = data_path
#         self.features = features

#     def load_dataset(self) -> DataLoader:
#         dataset = DatasetNLI(self.data_path, None)
#         dataloader = DataLoader(dataset, batch_size=1)
#         return dataloader

#     def collapse_dataset(self) -> DataLoader:
#         if self.features == None:
#             raise Exception("<Features not None>")
#         dataset = DatasetNLI(self.data_path, self.features)
#         dataloader = DataLoader(dataset, batch_size=1)
#         return dataloader
    
#     def preprocess(self) -> DataLoader:
#         pass