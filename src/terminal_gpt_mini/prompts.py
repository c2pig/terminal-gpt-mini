"""System prompts shared by the CLI and the tiny shell model.

Keep SYSTEM_PROMPT aligned with training data in ml/data. The generic client
path does not send this prompt.
"""

SYSTEM_PROMPT = (
    "You are a Linux terminal on Ubuntu. "
    "When the user message is a shell prompt of the form user@dev:<cwd>$ <command>, "
    "reply with only that command's stdout and stderr, exactly as the terminal would print it. "
    "Do not add explanations, markdown, or a new prompt. "
    "When the user asks a plain-English question, answer in one or two short sentences."
)


def system_prompt_for_model(model: str) -> str | None:
    """Return a system prompt only for the local shell model.

    Other model names keep the historical client behavior: no system message.
    """
    if "terminal-shell" in model:
        return SYSTEM_PROMPT
    return None
