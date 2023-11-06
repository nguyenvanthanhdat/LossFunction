import json
from abc import ABC, abstractclassmethod
from torch.utils.data import DataLoader, IterableDataset

class BaseDataset(ABC):
    @abstractclassmethod
    def load_dataset(self) -> DataLoader:
        # load sample dataset
        raise NotImplementedError

    @abstractclassmethod
    def collapse_dataset(self) -> DataLoader:
        # load sample dataset but less feature
        raise NotImplementedError

    @abstractclassmethod
    def transform_dataset(self) -> DataLoader:
        # Tokenize dataset
        raise NotImplementedError
    
    @abstractclassmethod
    def preprocess(self) -> DataLoader:
        # show how dataset is preprocessed
        raise NotImplementedError

class DatasetNLI(IterableDataset):
    def __init__(self, files, features):
        self.files = files
        self.features = features

    def __iter__(self):
        for json_file in self.files:
            with open(json_file) as f:
                for sample_line in f:
                    sample = json.loads(sample_line)
                    if self.features == None:
                        yield sample
                    else:
                        temp = []
                        for feature in self.features:
                            temp.append(sample[feature])
                        yield temp



class PreprocessDataset(IterableDataset):
    def __init__(self, data_collapse, preprocess_fn):
        self.data_collapse = data_collapse
        self.preprocess_fn = preprocess_fn

    def __iter__(self):
        for sample in self.data_collapse:
            sent1, sent2, label = sample
            new_sent, label = self.preprocess_fn(sent1[0], sent2[0], label[0])
            yield new_sent, label


class TokenizeDataset(IterableDataset):
    def __init__(self, data_preprocess, tokenizer):
        self.data_preprocess = data_preprocess
        self.tokenizer = tokenizer

    def tokenize(self, sent):
        new_sents = self.tokenizer(sent, return_tensors='pt')['input_ids']
        return new_sents[0]

    def __iter__(self):
        for sample in self.data_preprocess:
            sent, label = sample
            print(sent)
            new_sents = self.tokenize(sent[0])
            print(new_sents)
            # new_sents, label = self.tokenize(sent1[0], sent2[0], label)
            # print(new_sents)
            yield new_sents, label