from pathlib import Path
import chromadb
import hashlib
from sentence_transformers import SentenceTransformer


SOURCE_PATH = Path(__file__).parent

CHUNK_TOKENS = 350
OVERLAP_TOKENS = 50

MODEL_NAME = "intfloat/multilingual-e5-base"
model = SentenceTransformer(MODEL_NAME)
tok = model.tokenizer                     

STORE_PATH = SOURCE_PATH / "vectorstore"
client = chromadb.PersistentClient(path=str(STORE_PATH))
collection = client.get_or_create_collection(
    name="rag_corpus",
    configuration={"hnsw": {"space": "cosine"}},      # cosinus explicite
)

def deterministic_id(chunk: dict) -> str:
    key = f"{chunk['source']}::{chunk['chunk_index']}::{chunk['text']}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]

def load_documents(corpus_dir: Path) -> list[dict]:
    """
    Lit les fichiers .md et txt du dossier corpus_dir et retourne une liste de dictionnaires contenant le nom du fichier et son contenu.
    Si corpus_dir n'existe pas ou vide, lève une exception FileNotFoundError.

    Args:
        corpus_dir (Path): Le chemin vers le dossier contenant les fichiers à charger.
    Raises:
        FileNotFoundError: Si le dossier n'existe pas ou ne contient pas de fichiers .md ou .txt.
    Returns:
        list[dict]: {"source": <nom_du_fichier>, "text": <contenu>}.
    """
    if not corpus_dir.is_dir(): 
        raise FileNotFoundError(f"Le dossier {corpus_dir} n'existe pas.")

    files = sorted(
        p for p in corpus_dir.iterdir()
        if p.suffix.lower() in {".md", ".txt"}               # une passe, deux extensions, insensible à la casse
    )
    if not files:
        raise FileNotFoundError(f"Aucun fichier .md ou .txt dans {corpus_dir}.")

    return [
        {"source": p.name, "text": p.read_text(encoding="utf-8", errors="replace")}
        for p in files
    ]

def chunk_document(doc: dict) -> list[dict]:
    """
    Découpe le texte d'un document en chunks de taille CHUNK_TOKENS avec un chevauchement de OVERLAP_TOKENS.
    Retourne une liste de dictionnaires contenant le nom du fichier et le texte découpé.

    Args:
        doc (dict): Un dictionnaire contenant "source" et "text".
    Returns:
        list[dict]: {"text": str, "source": str, "chunk_index": int}.
    """
    tokens = tok.encode(doc["text"], add_special_tokens=False)
    step = CHUNK_TOKENS - OVERLAP_TOKENS
    chunks = []
    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i:i + CHUNK_TOKENS]
        chunk_text = tok.decode(chunk_tokens)
        chunks.append({
            "text": chunk_text, 
            "source": doc["source"], 
            "chunk_index": i // step
        })

    return chunks

def embed_passages(model, texts: list[str]) -> list[list[float]]:
    """
    Retourne les embeddings vectoriels pour une liste de textes.

    Args:
        model: Le modèle SentenceTransformer.
        texts (list[str]): Liste de textes à encoder.
    Returns:
        list[list[float]]: Liste d'embeddings vectoriels.
    """
    vectors = model.encode_document(texts, normalize_embeddings=True).tolist()
    return vectors

def main():
    documents = load_documents(SOURCE_PATH / "corpus")
    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
    print(f"Nombre total de chunks à indexer : {len(all_chunks)}")
    embeddings = embed_passages(model, [chunk["text"] for chunk in all_chunks])
    collection.add(
        ids=[deterministic_id(chunk) for chunk in all_chunks],          # list[str], identifiants UNIQUES
        documents=[chunk["text"] for chunk in all_chunks],    # list[str], le texte des chunks (pour relire et citer)
        embeddings=embeddings,                                # list[list[float]], tes vecteurs de A3
        metadatas=[{"source": chunk["source"], "chunk_index": chunk["chunk_index"]} for chunk in all_chunks],    # list[dict], {"source": ..., "chunk_index": ...}
    )
    print(f"Nombre total de chunks indexés : {collection.count()}")      # -> int

if __name__ == "__main__":
    main()

