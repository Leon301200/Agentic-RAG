from dataclasses import dataclass, field

@dataclass
class Trace:
    steps: int = 0
    tool_calls: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    outcome: str = "unknown" 

    def cost_usd(self, in_price=0.80, out_price=0.80) -> float:
        # tarifs Haiku 4.5 en $/million de tokens — à vérifier sur claude.com/pricing
        return (self.input_tokens * in_price + self.output_tokens * out_price) / 1e6

    def __str__(self) -> str:
        return (
            f"Trace(steps={self.steps}, tool_calls={self.tool_calls}, "
            f"input_tokens={self.input_tokens}, output_tokens={self.output_tokens}, "
            f"outcome='{self.outcome}', cost_usd={self.cost_usd():.6f})"
        )