"""System prompts shared by the CLI and the tiny shell model.

Keep SYSTEM_PROMPT aligned with training data in ml/data. The generic client
path does not send this prompt.
"""

SYSTEM_PROMPT = (
    "You are a Linux terminal on Ubuntu. "
    "If the user message is a shell prompt and command, reply with only the command output. "
    "If the user asks a question, answer in one or two short sentences."
)


def system_prompt_for_model(model: str) -> str | None:
    """Return a system prompt only for the local shell model.

    Other model names keep the historical client behavior: no system message.
    """
    if "terminal-shell" in model:
        return SYSTEM_PROMPT
    return None
