import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

EMBED_MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def generate_features(excerpt: str) -> list:
    sentences = sent_tokenize(excerpt)

    # mean excerpt embedding
    sent_embeddings = EMBED_MODEL.encode(sentences)
    mean_embedding = sent_embeddings.mean(axis=0)
    mean_embedding = list(mean_embedding)

    excerpt_features = mean_embedding
    return excerpt_features