from transformers import pipeline
from typing import List


class ModelFacade:
    def __init__(self, model_name: str):
        self.summarizer = pipeline("summarization", model=model_name)

    def __call__(self, texts: List[str], min_length: int = 10, max_length: int = 50) -> List[str]:
        try:
            return [i["summary_text"] for i in self.summarizer(texts, max_length=max_length, min_length=min_length)]
        except IndexError:
            texts = [" ".join(i.split()[:-50]) for i in texts]
            return self.__call__(texts, min_length, max_length)


class GooglePegasus(ModelFacade):
    def __init__(self):
        super().__init__("google/pegasus-xsum")

    def __call__(self, texts: List[str], min_length: int = 10, max_length: int = 50) -> List[str]:
        texts = [" ".join(i.split()[:512]) for i in texts]
        return super().__call__(texts, min_length, max_length)


class Toloka(ModelFacade):
    def __init__(self):
        super().__init__("toloka/t5-large-for-text-aggregation")

    def __call__(self, texts: List[str], min_length: int = 10, max_length: int = 50) -> List[str]:
        texts = [" ".join(i.split()[:512]) for i in texts]
        return super().__call__(texts, min_length, max_length)


class Bart(ModelFacade):
    def __init__(self):
        super().__init__("facebook/bart-large-cnn")

    def __call__(self, texts: List[str], min_length: int = 10, max_length: int = 50) -> List[str]:
        texts = [" ".join(i.split()[:1024]) for i in texts]
        return super().__call__(texts, min_length, max_length)

