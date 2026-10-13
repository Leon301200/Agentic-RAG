from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from embeddings import get_model

SOURCE_PATH = Path(__file__).parent

model = get_model()
tok = model.tokenizer

STORE_PATH = SOURCE_PATH / "vectorstore"
client = chromadb.PersistentClient(path=str(STORE_PATH))
collection = client.get_or_create_collection(
    name="rag_corpus",
    configuration={"hnsw": {"space": "cosine"}},      # cosinus explicite
)

def search(query: str, k: int = 5, where: dict | None = None) -> list[dict]:
    """
    Cherche les k chunks les plus proches de la requête dans la collection.

    Args:
        query (str): La requête en langage naturel (le préfixe "query: " est ajouté à l'encodage).
        k (int): Nombre de chunks à retourner.
        where (dict | None): Filtre optionnel sur les métadonnées, ex. {"source": "..."}.
    Returns:
        list[dict]: {"id": str, "text": str, "source": str, "chunk_index": int, "score": float},
        par ordre décroissant de score (similarité cosinus).
    """
    vector = model.encode_query(query, normalize_embeddings=True).tolist()   # préfixe "query: " via embeddings.py
    results = collection.query(
        query_embeddings=[vector],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],   # explicite : le défaut varie selon la version de Chroma
    )

    ids = results["ids"][0]              # list[str]   (les k identifiants)
    documents = results["documents"][0]  # list[str]   (les k textes)
    metadatas = results["metadatas"][0]  # list[dict]  (les k métadonnées)
    distances = results["distances"][0]  # list[float] (distances cosinus, PAS des similarités)

    chunks = [
        {
            "id": id_,
            "text": doc,
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "score": 1 - dist,                             # distance cosinus -> similarité cosinus
        }
        for id_, doc, meta, dist in zip(ids, documents, metadatas, distances)
    ]
    return sorted(chunks, key=lambda c: c["score"], reverse=True)

def format_results(chunks: list[dict]) -> str:
    """
    Met en forme les chunks retournés par search() en extraits numérotés et sourcés.

    Args:
        chunks (list[dict]): La sortie de search().
    Returns:
        str: Les extraits au format "[i] (source)\ntexte", ou un message si la liste est vide.
    """
    if not chunks:
        return "Aucun document trouvé pour cette requête."

    return "\n\n".join(
        f"[{i}] ({chunk['source']})\n{chunk['text'][:1500]}"
        for i, chunk in enumerate(chunks, 1)
    )
