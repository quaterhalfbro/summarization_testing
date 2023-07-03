from torch.utils.data import Dataset, DataLoader
from typing import List


class SummarizationDataset(Dataset):
    def __init__(self, x: List[str], y: List[str]):
        super().__init__()
        assert len(x) == len(y)
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return {
            "text": self.x[item],
            "summarized_text": self.y[item]
        }


class SummarizationDataLoader(DataLoader):
    def __init__(self, x: List[str], y: List[str], batch_size: int):
        super().__init__(SummarizationDataset(x, y), batch_size=batch_size)
