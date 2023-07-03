import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


def bleu(reference: str, hypothesis: str) -> float:
    reference_tokens = nltk.tokenize.word_tokenize(reference.lower())
    hypothesis_tokens = nltk.tokenize.word_tokenize(hypothesis.lower())
    return sentence_bleu([reference_tokens], hypothesis_tokens)


def meteor(reference: str, hypothesis: str) -> float:
    reference_tokens = nltk.tokenize.word_tokenize(reference.lower())
    hypothesis_tokens = nltk.tokenize.word_tokenize(hypothesis.lower())
    return meteor_score([reference_tokens], hypothesis_tokens)


class EmbeddingScore:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def calculate(self, a: str, b: str) -> float:
        encoded_input = self.tokenizer([a, b], padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = EmbeddingScore.mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return F.cosine_similarity(sentence_embeddings[0], sentence_embeddings[1], dim=0).tolist()



