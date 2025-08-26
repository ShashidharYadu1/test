import logging
from datetime import datetime
from typing import Dict, Optional


class LLMUsageTracker:
    PRICING = {
        "claude-3-5-sonnet-20241022": {
            "input": 0.003,
            "output": 0.015,
        },
    }

    def __init__(self, log_file: str = "llm_usage.log"):
        self.logger = logging.getLogger("LLMUsageLogger")
        self.logger.setLevel(logging.INFO)

        # Only add handler once (important for repeated use)
        if not self.logger.handlers:
            file_handler = logging.FileHandler(log_file, mode="a")  # 'a' = append
            formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def calculate_cost(
        self, input_tokens: int, output_tokens: int, model: str
    ) -> float:
        if model not in self.PRICING:
            raise ValueError(f"Unknown model: {model}")
        input_cost = (input_tokens / 1000) * self.PRICING[model]["input"]
        output_cost = (output_tokens / 1000) * self.PRICING[model]["output"]
        return input_cost + output_cost

    def track_llm_call(
        self, response: Dict, model: str, file_name: Optional[str] = None, response_time = float) -> Dict:
        usage = response.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        total_tokens = input_tokens + output_tokens
        cost = self.calculate_cost(input_tokens, output_tokens, model)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        log_data = {
            "timestamp": timestamp,
            "model": model,
            "file_name": file_name or "N/A",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_token": total_tokens,
            "response time":response_time,
            "cost": round(cost, 4),
        }

        self.logger.info(f"LLM Usage: {log_data}")
        return log_data