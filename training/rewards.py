import json
import re

def parse_model_output(text: str):
    """
    Sniper Parser: Finds the FIRST valid JSON object using bracket balancing.
    Ignores any 'yapping' after the closing brace.
    """
    # 1. Cleaning
    clean_text = text.replace("```json", "").replace("```", "").strip()

    # 2. Find the FIRST '{'
    start_index = clean_text.find('{')
    if start_index == -1:
        return None, False

    # 3. Walk forward and count brackets to find the matching '}'
    balance = 0
    is_inside_string = False
    escape_next = False
    
    for i in range(start_index, len(clean_text)):
        char = clean_text[i]
        
        # Handle string content (ignore brackets inside strings)
        if char == '"' and not escape_next:
            is_inside_string = not is_inside_string
        
        if not is_inside_string:
            if char == '{':
                balance += 1
            elif char == '}':
                balance -= 1
        
        # Handle escape characters (e.g. \")
        if char == '\\' and not escape_next:
            escape_next = True
        else:
            escape_next = False

        # If balance hits zero, we found the closing brace
        if balance == 0:
            candidate = clean_text[start_index : i+1]
            try:
                return json.loads(candidate), True
            except:
                # If strictly parsing fails, it's not clean
                break
    
    # 4. Fallback (Regex Rescue)
    # If the strict loop failed or didn't finish, try the regex grab
    match = re.search(r'"question":\s*"(.*?)"', text, re.DOTALL)
    if match:
        return {
            "question": match.group(1),
            "options": ["A", "B", "C", "D"],
            "answer": "A",
            "explanation": "Regex Rescued"
        }, False
    
    return None, False

def format_reward_func(completions, **kwargs):
    """
    Reward 1.0 for valid JSON.
    Reward 0.1 for Regex Rescue.
    """
    rewards = []
    for content in completions:
        data, is_clean = parse_model_output(content)
        if data and is_clean:
            if all(k in data for k in ["question", "options", "answer", "explanation"]):
                rewards.append(1.0)
            else:
                rewards.append(0.5)
        elif data and not is_clean:
            rewards.append(0.1) # Penalize "Rescue" cases
        else:
            rewards.append(0.0)
    return rewards

def complexity_reward_func(completions, **kwargs):
    """
    Reward complexity, but ONLY on the valid parsed data.
    """
    rewards = []
    keywords = ["__del__", "cycle", "weakref", "GIL", "gc.collect", "leak", "ref count"]
    
    for content in completions:
        score = 0.0
        data, _ = parse_model_output(content) # Use the same parser!
        
        if not data:
            rewards.append(0.0)
            continue
            
        # Analyze the GENERATED question text
        q_text = data.get("question", "")
        exp_text = data.get("explanation", "")
        full_gen = q_text + exp_text
        
        # Length Bonus
        if len(full_gen) > 150: score += 0.2
        if len(full_gen) > 300: score += 0.3
        
        # Keyword Bonus
        hits = sum(1 for k in keywords if k in full_gen)
        score += min(hits * 0.1, 0.5)
        
        rewards.append(score)
        
    return rewards