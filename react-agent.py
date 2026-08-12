import os
import json
import anthropic
from tracing import Trace
from pydantic import ValidationError
from tools import REGISTRY, schemas

client = anthropic.Client(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL = "claude-haiku-4-5-20251001"

# Les implémentations

SYSTEM = (
    "Tu es un assistant qui résout des questions étape par étape. "
    "Avant chaque appel d'outil, explique en une phrase pourquoi tu l'appelles. "
    "N'invente jamais un chiffre : si un outil peut te le donner, appelle-le."
)

# L'exécuteur (isolé : il ne doit JAMAIS lever d'exception)

def execute_tool(name: str, raw_input: dict) -> str:
    """Renvoie (contenu, is_error). Une erreur est une OBSERVATION, pas un crash."""
    fn = REGISTRY.get(name)
    if fn is None:
        return f"Outil inconnu : '{name}'. Disponible : {list(REGISTRY)}", True
    try:
        return fn.run(raw_input), False
    except ValidationError as e:
        return f"Arguments invalides pour '{name}' : {e.errors()}", True
    except Exception as e:
        return f"{type(e).__name__}: {e}", True

# La Boucle

def run(question: str, max_steps: int = 5, trace: Trace | None = None, verbose: bool = False) -> str:
    
    if trace is None:
        trace = Trace()

    messages = [
        {"role": "user", "content": question},
    ]

    for step in range(1, max_steps + 1):

        trace.steps += 1

        resp = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM,
            tools=schemas(),
            messages=messages,
        )

        trace.input_tokens += resp.usage.input_tokens
        trace.output_tokens += resp.usage.output_tokens

        # Garde fou : réponse tronquée
        if resp.stop_reason == "max_tokens":
            trace.outcome = "max_tokens"
            return "[échec] réponse tronquée (max_tokens) — augmente la limite"

        messages.append({"role": "assistant", "content": resp.content})

        if verbose:
            for block in resp.content:
                if block.type == "text":
                    print(f"\n🧠 [{step}] Thought: {block.text.strip()}")
                elif block.type == "tool_use":
                    print(f"\n🔧 [{step}] Action : {block.name}({json.dumps(block.input, ensure_ascii=False)})")

        # Si l'assistant a fini, on retourne la réponse finale
        if resp.stop_reason != "tool_use":
            final = "".join(b.text for b in resp.content if b.type == "text")
            if verbose:
                print(f"\n📊 tokens: {trace.input_tokens} in / {trace.output_tokens} out")
            trace.outcome = "answered"
            return final.strip()

        # Sinon, on exécute le ou les outils demandé(s)
        results = []
        for block in resp.content:
            if block.type != 'tool_use':
                continue
            tool_name = block.name
            tool_input = block.input
            content, is_error = execute_tool(tool_name, tool_input)

            trace.tool_calls.append(tool_name)

            if verbose:
                print(f"👁️  [{step}] Observation: {content}{' [ERREUR]' if is_error else ''}")

            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": content,
                "is_error": is_error,
            })

        messages.append({"role": "user", "content": results})

    trace.outcome = "max_steps"
    return f"[échec] plafond de {max_steps} itérations atteint sans réponse finale"

if __name__ == "__main__":
    questions = [
        ("Quel est le délai de rétractation pour un achat en ligne ? "
         "Le terme « délai de rétractation » est présent tel quel dans le corpus. Un chunk, réponse directe (14 jours)."
         "C'est ton cas de contrôle « facile » — et c'est la seule des six à laquelle ton stub actuel répond déjà."),
    ]

    for q in questions:
        print("\n=== QUESTION ===\n" + q)
        trace = Trace()
        print("\n=== RÉPONSE ===\n" + run(q, max_steps=5, trace=trace, verbose=True))
        print("\n=== TRACE ===\n" + str(trace))