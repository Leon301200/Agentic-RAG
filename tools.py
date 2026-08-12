from dataclasses import dataclass
from typing import Callable
from pydantic import BaseModel, Field

REGISTRY: dict[str, "Tool"] = {}

@dataclass
class Tool:
    name: str
    description: str
    args: type[BaseModel]
    fn: Callable

    def schema(self) -> dict:
        s = self.args.model_json_schema()
        s.pop("title", None)
        return {"name": self.name, "description": self.description, "input_schema": s}

    def run(self, raw: dict) -> str:
        """Exécute l'outil avec les arguments fournis."""
        validated_args = self.args(**raw)
        return self.fn(**validated_args.model_dump())

def register_tool(name: str, description: str, args: type[BaseModel]):
    """Décorateur pour enregistrer un outil."""
    def decorator(fn: Callable):
        REGISTRY[name] = Tool(name=name, description=description, args=args, fn=fn)
        return fn
    return decorator

def schemas() -> list[dict]:
    """Renvoie la liste des outils enregistrés sous forme de schéma JSON."""
    return [tool.schema() for tool in REGISTRY.values()]


# OUTIL 1 : CALCULATRICE ------------------------
import ast, operator

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}

def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        if isinstance(node.op, ast.Pow):                       # garde anti-DoS
            exp = _eval(node.right)
            if abs(exp) > 100:
                raise ValueError("Exposant trop grand (max 100).")
            return _OPS[ast.Pow](_eval(node.left), exp)
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError(f"Expression non autorisée : {type(node).__name__}")

class CalcArgs(BaseModel):
    expression: str = Field(
        description="Expression arithmétique pure, ex: '(2133111 - 522250) / 522250 * 100'",
        max_length=200,
    )

@register_tool(
    name="calculator",
    description=(
        "Évalue une expression arithmétique et renvoie le résultat exact. "
        "Utilise-le pour TOUT calcul, même simple — ne calcule jamais de tête. "
        "Opérateurs autorisés : + - * / // % ** et parenthèses. Ni variables, ni fonctions, ni texte."
    ),
    args=CalcArgs
)
def calculator(expression: str) -> str:
    result = _eval(ast.parse(expression, mode="eval").body)
    return str(round(result, 6) if isinstance(result, float) else result)


# OUTIL 2 : RECHERCHE WEB ------------------------
from ddgs import DDGS

class SearchArgs(BaseModel):
    query: str = Field(description="Requête en mots-clés, 3 à 8 mots. Pas de phrase complète.")
    max_results: int = Field(default=3, ge=1, le=5, description="Nombre de résultats (1-5).")

@register_tool(
        name="web_search",
        description=(
            "Recherche sur le web et renvoie des extraits de résultats (titre, URL, extrait). "
            "Utilise-le pour des faits récents, des chiffres actualisés ou toute information "
            "que tu ne connais pas avec certitude. Ne l'utilise PAS pour du calcul ni pour lire "
            "un fichier local. Une requête = un sujet ; découpe les questions composées."
        ),
        args=SearchArgs
)
def web_search(query: str, max_results: int) -> str:
    with DDGS() as ddgs:
        hits = list(ddgs.text(query, max_results=max_results))
    if not hits:
        return "Aucun résultat. Reformule la requête avec des mots-clés différents."
    return "\n\n".join(
        f"[{i}] {h['title']}\n{h['href']}\n{h['body'][:400]}"
        for i, h in enumerate(hits, 1)
    )


# OUTIL 3 : LECTURE DE FICHIER LOCAL ------------------------
from pathlib import Path

SANDBOX = Path("./workspace").resolve()      # ← LA frontière
MAX_CHARS = 8000
ALLOWED = {".txt", ".md", ".csv", ".json", ".py", ".log"}

class ReadFileArgs(BaseModel):
    path: str = Field(description="Chemin RELATIF au workspace, ex: 'notes/reunion.md'")

@register_tool(
    name="read_file",
    description=(
        "Lit un fichier texte du workspace local et renvoie son contenu. "
        "Utilise-le quand la question porte sur un document de l'utilisateur. "
        "Le chemin doit être relatif au workspace ; les chemins absolus et les '..' sont refusés. "
        "Si tu ne connais pas le nom du fichier, tente 'index.md' : l'erreur listera les fichiers disponibles."
    ),
    args=ReadFileArgs
)
def read_file(path: str) -> str:
    target = (SANDBOX / path).resolve()

    if not target.is_relative_to(SANDBOX):                       # ← anti path-traversal
        raise ValueError(f"Accès refusé : '{path}' sort du workspace autorisé.")
    if not target.is_file():
        dispo = sorted(p.relative_to(SANDBOX).as_posix() for p in SANDBOX.rglob("*") if p.is_file())
        raise FileNotFoundError(f"'{path}' introuvable. Fichiers disponibles : {dispo}")
    if target.suffix.lower() not in ALLOWED:
        raise ValueError(f"Extension non autorisée : {target.suffix}. Autorisées : {sorted(ALLOWED)}")

    text = target.read_text(encoding="utf-8", errors="replace")
    if len(text) > MAX_CHARS:
        return text[:MAX_CHARS] + f"\n\n[... tronqué — {len(text)} caractères au total]"
    return text

# OUTIL 4 : RECHERCHE DE DOCUMENTS ------------------------

class SearchDocsArgs(BaseModel):
    query: str = Field(description="Question ou mots-clés décrivant l'information recherchée dans le corpus.")
    max_results: int = Field(default=3, ge=1, le=5, description="Nombre d'extraits (chunks) à renvoyer (1-5).")

@register_tool(
        name="search_documents",
        description=(
            "Recherche dans le corpus documentaire de référence et renvoie des extraits "
            "pertinents, numérotés et sourcés. Utilise-le quand la question porte sur le contenu "
            "du corpus (réglementation, jurisprudence, documents fournis). Appuie ta réponse sur "
            "ces extraits et cite leurs numéros, ex. [1], [2]. "
            "Différent de read_file (qui lit UN fichier nommé) et de web_search (le web externe)."
        ),
        args=SearchDocsArgs
)
def search_documents(query: str, max_results:int) -> str:
    chunks = [
        {"source": "reglement_A.pdf", "article": "art. 12",
        "text": "Le consommateur dispose de quatorze jours pour se rétracter."},
        {"source": "reglement_B.pdf", "article": "art. 9",
        "text": "Un délai de rétractation de 14 jours s'applique aux ventes à distance."},
        {"source": "reglement_C.pdf", "article": "art. 3",
        "text": "Les conditions de retour doivent être clairement indiquées."},
    ]
    return "\n\n".join(
        f"[{i}] ({chunk['source']}, {chunk['article']}) {chunk['text']}"
        for i, chunk in enumerate(chunks[:max_results], 1)
    )