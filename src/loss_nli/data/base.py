import json
from abc import ABC, abstractclassmethod
from datasets import DatasetDict, Dataset, load_from_disk
from typing import Tuple

class BaseDataset(ABC): 
    @abstractclassmethod
    def load_from_disk(self) -> DatasetDict: 
        raise NotImplementedError
    
    @abstractclassmethod
    def tokenize(self) -> Tuple[Dataset]:
        raise NotImplementedError
    
    @abstractclassmethod
    def save_disk(self): 
        raise NotImplementedError
    
    
    @abstractclassmethod
    def get_dataset(self) -> Tuple[Dataset]:
        raise NotImplementedError



# class BaseDataset(ABC):
#     @abstractclassmethod
#     def load_dataset(self) -> DataLoader:
#         # load sample dataset
#         raise NotImplementedError

#     @abstractclassmethod
#     def collapse_dataset(self) -> DataLoader:
#         # load sample dataset but less feature
#         raise NotImplementedError

#     @abstractclassmethod
#     def transform_dataset(self) -> DataLoader:
#         # Tokenize dataset
#         raise NotImplementedError
    
#     @abstractclassmethod
#     def preprocess(self) -> DataLoader:
#         # show how dataset is preprocessed
#         raise NotImplementedError

# class DatasetNLI(IterableDataset):
#     def __init__(self, files, features):
#         self.files = files
#         self.features = features

#     def __len__(self):
#         length = 0
#         for json_file in self.files:
#             with open(json_file) as f:
#                 length += len(f.readlines())
#         return length

#     def __iter__(self):
#         for json_file in self.files:
#             with open(json_file) as f:
#                 for sample_line in f:
#                     sample = json.loads(sample_line)
#                     if self.features == None:
#                         yield sample
#                     else:
#                         temp = []
#                         for feature in self.features:
#                             temp.append(sample[feature])
#                         yield temp



# class PreprocessDataset(IterableDataset):
#     def __init__(self, data_collapse, preprocess_fn):
#         self.data_collapse = data_collapse
#         self.preprocess_fn = preprocess_fn

#     def __len__(self):
#         return(len(self.data_collapse))

#     def __iter__(self):
#         for sample in self.data_collapse:
#             sent1, sent2, label = sample
#             new_sent, label = self.preprocess_fn(sent1[0], sent2[0], label[0])
#             yield new_sent, label


# class TokenizeDataset(IterableDataset):
#     def __init__(self, data_preprocess, tokenizer, max_length):
#         self.data_preprocess = data_preprocess
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#     def __len__(self):
#         return(len(self.data_preprocess))

#     def __iter__(self):
#         for sample in self.data_preprocess:
#             sent, label = sample
#             input_ids = self.tokenizer.encode(sent[0],
#                                               max_length=self.max_length,
#                                               padding="max_length",
#                                               truncation=True,
#                                               return_tensors='pt')[0]
#             label = label[0]
#             yield input_ids, label