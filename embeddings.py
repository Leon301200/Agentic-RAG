from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-base"

def get_model() -> SentenceTransformer:
    return SentenceTransformer(
        MODEL_NAME,
        prompts={"query": "query: ", "document": "passage: "},
    )