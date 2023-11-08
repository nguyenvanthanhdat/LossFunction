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