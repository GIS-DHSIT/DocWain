def build_dpo_prompt(query: str, model_response: str, scores: dict) -> str:
    score_text = "\n".join(f"  - {k}: {v:.2f}/1.00" for k, v in scores.items())
    return f"""You are evaluating and improving a DocWain response.

User query: "{query}"

DocWain's response:
---
{model_response}
---

Current scores (0-1 scale):
{score_text}

Tasks:
1. Identify the specific weaknesses in this response
2. Generate a BETTER response that fixes these weaknesses

Output format:
WEAKNESSES: [brief list of issues]
---IMPROVED---
[your improved response here]"""
