class PipelineGuard:
    def __init__(self, max_steps: int = 8, cost_cap_usd: float = 1.00):
        self.max_steps = max_steps
        self.cost_cap_usd = cost_cap_usd
        self.steps_used = 0
        self.cost_used = 0.0

    def step(self, name: str, estimated_cost: float = 0.0):
        self.steps_used += 1
        self.cost_used += estimated_cost

        if self.steps_used > self.max_steps:
            raise RuntimeError(
                f"Max steps exceeded: used {self.steps_used}, cap {self.max_steps}"
            )

        if self.cost_used > self.cost_cap_usd:
            raise RuntimeError(
                f"Cost cap exceeded: used {self.cost_used:.4f}, cap {self.cost_cap_usd:.4f}"
            )

        return {
            "step_name": name,
            "steps_used": self.steps_used,
            "cost_used": round(self.cost_used, 4),
        }