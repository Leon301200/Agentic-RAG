from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

SOURCE_PATH = Path(__file__).parent

MODEL_NAME = "intfloat/multilingual-e5-base"
model = SentenceTransformer(MODEL_NAME)
tok = model.tokenizer        

STORE_PATH = SOURCE_PATH / "vectorstore"
client = chromadb.PersistentClient(path=str(STORE_PATH))
collection = client.get_or_create_collection(
    name="rag_corpus",
    configuration={"hnsw": {"space": "cosine"}},      # cosinus explicite
)

def search_documents(query: str, k: int = 5) -> str:
    text = "query: " + query
    vector = model.encode(text, normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=[vector],
        n_results=k,
        # where={"source": "..."},                  # filtre optionnel sur les métadonnées (tu t'en sers en B4)
    )

    ids = results["ids"][0]              # list[str]   (les k identifiants)
    documents = results["documents"][0]  # list[str]   (les k textes)
    metadatas = results["metadatas"][0]  # list[dict]  (les k métadonnées)

    if not documents:
        return "Aucun document trouvé pour cette requête."
    
    return "\n\n".join(
        f"[{i}] ({meta['source']})\n{doc[:1500]}"
        for i, (doc, meta) in enumerate(zip(documents, metadatas), 1)
    )

print(search_documents("délai de rétractation", k=5))