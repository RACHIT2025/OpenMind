"""
OpenMind Chat Templates.

Handles formatting conversations into prompt/completion pairs for:
- Supervised Fine-Tuning (SFT)
- Direct Preference Optimization (DPO)

Supports multiple dataset formats:
- Alpaca (instruction/input/output)
- ShareGPT (multi-turn conversations)
- Anthropic HH-RLHF (chosen/rejected)
"""

from typing import Optional


# ─── Chat Template ────────────────────────────────────────────────────────────

CHAT_TEMPLATE = """<|system|>
{system_message}<|endoftext|>
<|user|>
{user_message}<|endoftext|>
<|assistant|>
{assistant_message}<|endoftext|>"""

SYSTEM_DEFAULT = "You are OpenMind, a helpful, harmless, and honest AI assistant."


def format_chat(
    messages: list[dict],
    system_prompt: str = SYSTEM_DEFAULT,
    add_generation_prompt: bool = False,
    template: str = "chat",
) -> str:
    """
    Format a list of chat messages into a template.

    Args:
        messages: List of {"role": "user"|"assistant"|"system", "content": "..."}
        system_prompt: Default system prompt if none in messages
        add_generation_prompt: If True, add assistant prompt for generation
        template: Template format to use: "chat", "alpaca", or "raw"

    Returns:
        Formatted text string
    """
    if template == "alpaca":
        parts = []
        # Find system message or use default
        system_content = system_prompt
        for m in messages:
            if m["role"] == "system":
                system_content = m["content"]
        
        parts.append(f"{system_content}\n\nBelow is an instruction that describes a task. Write a response that appropriately completes the request.")
        
        for msg in messages:
            if msg["role"] == "user":
                parts.append(f"\n### Instruction:\n{msg['content']}")
            elif msg["role"] == "assistant":
                parts.append(f"\n### Response:\n{msg['content']}")
                
        if add_generation_prompt:
            parts.append("\n### Response:\n")
            
        return "\n".join(parts)
        
    elif template == "raw":
        return "\n".join(msg["content"] for msg in messages)
        
    else:
        # Default "chat" template
        parts = []

        # Check if system message is in messages
        has_system = any(m["role"] == "system" for m in messages)
        if not has_system:
            parts.append(f"<|system|>\n{system_prompt}<|endoftext|>")

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|{role}|>\n{content}<|endoftext|>")

        if add_generation_prompt:
            parts.append("<|assistant|>\n")

        return "\n".join(parts)



# ─── Dataset Format Parsers ───────────────────────────────────────────────────

def parse_alpaca(example: dict) -> dict:
    """
    Parse Alpaca-format example.

    Input format: {"instruction": "...", "input": "...", "output": "..."}
    """
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    if input_text:
        user_message = f"{instruction}\n\nInput: {input_text}"
    else:
        user_message = instruction

    messages = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": output},
    ]

    text = format_chat(messages)
    return {"text": text, "messages": messages}


def parse_sharegpt(example: dict) -> dict:
    """
    Parse ShareGPT-format example.

    Input format: {"conversations": [{"from": "human", "value": "..."}, ...]}
    """
    conversations = example.get("conversations", [])
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}

    messages = []
    for turn in conversations:
        role = role_map.get(turn.get("from", ""), "user")
        content = turn.get("value", "")
        messages.append({"role": role, "content": content})

    text = format_chat(messages)
    return {"text": text, "messages": messages}


def parse_hh_rlhf(example: dict) -> dict:
    """
    Parse Anthropic HH-RLHF format for DPO training.

    Input format: {"chosen": "Human: ... Assistant: ...", "rejected": "Human: ... Assistant: ..."}
    Returns: {"prompt": "...", "chosen": "...", "rejected": "..."}
    """
    def extract_turns(text: str) -> list[dict]:
        messages = []
        parts = text.split("\n\nHuman: ")
        for i, part in enumerate(parts):
            if not part.strip():
                continue
            if "Assistant: " in part:
                human_part, assistant_part = part.split("\n\nAssistant: ", 1)
                if i == 0:
                    human_part = human_part.replace("Human: ", "", 1)
                messages.append({"role": "user", "content": human_part.strip()})
                messages.append({"role": "assistant", "content": assistant_part.strip()})
            else:
                clean = part.replace("Human: ", "", 1) if i == 0 else part
                messages.append({"role": "user", "content": clean.strip()})
        return messages

    chosen_messages = extract_turns(example.get("chosen", ""))
    rejected_messages = extract_turns(example.get("rejected", ""))

    # Prompt is everything except the last assistant response
    prompt_messages = chosen_messages[:-1] if chosen_messages else []
    prompt = format_chat(prompt_messages, add_generation_prompt=True)

    chosen_response = chosen_messages[-1]["content"] if chosen_messages else ""
    rejected_response = rejected_messages[-1]["content"] if rejected_messages else ""

    return {
        "prompt": prompt,
        "chosen": chosen_response,
        "rejected": rejected_response,
    }


FORMAT_PARSERS = {
    "alpaca": parse_alpaca,
    "sharegpt": parse_sharegpt,
    "hh_rlhf": parse_hh_rlhf,
}


def get_parser(format_name: str):
    """Get the parser function for a given format name."""
    if format_name not in FORMAT_PARSERS:
        raise ValueError(
            f"Unknown format: {format_name}. Available: {list(FORMAT_PARSERS.keys())}"
        )
    return FORMAT_PARSERS[format_name]
