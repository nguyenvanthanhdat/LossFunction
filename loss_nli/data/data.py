# from .base import BaseDataset, DatasetNLI
from .base import *

class ViNLI(BaseDataset):
    def __init__(self, data_path, preprocess_fn, tokenizer, features=None):
        if not isinstance(data_path, list):
            raise TypeError("Change <data_path> to list. If has 1 path please format \"[<data_path>]\"")
        self.data_path = data_path
        self.features = features
        self.preprocess_fn = preprocess_fn
        self.tokenizer = tokenizer

    def load_dataset(self) -> DataLoader:
        dataset = DatasetNLI(self.data_path, None)
        dataloader = DataLoader(dataset, batch_size=1)
        return dataloader

    def collapse_dataset(self) -> DataLoader:
        if self.features == None:
            raise Exception("<Features not None>")
        dataset = DatasetNLI(self.data_path, self.features)
        dataloader = DataLoader(dataset, batch_size=1)
        return dataloader
    
    def preprocess(self) -> DataLoader:
        dataset = PreprocessDataset(self.collapse_dataset(), self.preprocess_fn)
        dataloader = DataLoader(dataset, batch_size=64)
        return dataloader

    def transform_dataset(self) -> DataLoader:
        dataset = PreprocessDataset(self.collapse_dataset(), self.preprocess_fn)
        dataloader = DataLoader(dataset, batch_size=1)
        new_dataset = TokenizeDataset(dataloader, self.tokenizer)
        new_dataloader = DataLoader(new_dataset, batch_size=4)
        return new_dataloader
    
class SNLI(BaseDataset):
    def __init__(self, data_path, features=None):
        if not isinstance(data_path, list):
            raise TypeError("Change <data_path> to list. If has 1 path please format \"[<data_path>]\"")
        self.data_path = data_path
        self.features = features

    def load_dataset(self) -> DataLoader:
        dataset = DatasetNLI(self.data_path, None)
        dataloader = DataLoader(dataset, batch_size=1)
        return dataloader

    def collapse_dataset(self) -> DataLoader:
        if self.features == None:
            raise Exception("<Features not None>")
        dataset = DatasetNLI(self.data_path, self.features)
        dataloader = DataLoader(dataset, batch_size=1)
        return dataloader
    
    def preprocess(self) -> DataLoader:
        pass

class MultiNLI(BaseDataset):
    def __init__(self, data_path, features=None):
        if not isinstance(data_path, list):
            raise TypeError("Change <data_path> to list. If has 1 path please format \"[<data_path>]\"")
        self.data_path = data_path
        self.features = features

    def load_dataset(self) -> DataLoader:
        dataset = DatasetNLI(self.data_path, None)
        dataloader = DataLoader(dataset, batch_size=1)
        return dataloader

    def collapse_dataset(self) -> DataLoader:
        if self.features == None:
            raise Exception("<Features not None>")
        dataset = DatasetNLI(self.data_path, self.features)
        dataloader = DataLoader(dataset, batch_size=1)
        return dataloader
    
    def preprocess(self) -> DataLoader:
        pass

class ContractNLI(BaseDataset):
    def __init__(self, data_path, features=None):
        if not isinstance(data_path, list):
            raise TypeError("Change <data_path> to list. If has 1 path please format \"[<data_path>]\"")
        self.data_path = data_path
        self.features = features

    def load_dataset(self) -> DataLoader:
        dataset = DatasetNLI(self.data_path, None)
        dataloader = DataLoader(dataset, batch_size=1)
        return dataloader

    def collapse_dataset(self) -> DataLoader:
        if self.features == None:
            raise Exception("<Features not None>")
        dataset = DatasetNLI(self.data_path, self.features)
        dataloader = DataLoader(dataset, batch_size=1)
        return dataloader
    
    def preprocess(self) -> DataLoader:
        pass