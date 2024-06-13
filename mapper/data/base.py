from abc import abstractmethod
from typing import Optional


class DataBase():
    def __init__(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def prepare_data(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def setup(self, stage: Optional[str] = None):
        raise NotImplementedError

    @abstractmethod
    def dataset(self, stage: str):
        raise NotImplementedError