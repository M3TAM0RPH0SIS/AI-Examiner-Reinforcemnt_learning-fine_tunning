import json
import re

def format_reward_func(completions, **kwargs):
    """Reward 1.0 if strict JSON with all keys, 0.5 if partial."""
    rewards = []
    for content in completions:
        try:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if not match:
                rewards.append(0.0)
                continue
            data = json.loads(match.group(0))
            if all(k in data for k in ["question", "options", "answer", "explanation"]):
                rewards.append(1.0)
            else:
                rewards.append(0.5)
        except:
            rewards.append(0.0)
    return rewards

def complexity_reward_func(completions, **kwargs):
    """Reward longer code snippets and trap keywords."""
    rewards = []
    keywords = ["__del__", "cycle", "weakref", "GIL", "gc.collect", "leak"]
    for content in completions:
        score = 0.0
        # Length Bonus
        if len(content) > 200: score += 0.2
        if len(content) > 400: score += 0.3
        # Trap Bonus
        hits = sum(1 for k in keywords if k in content)
        score += (hits * 0.1)
        rewards.append(score)
    return rewards