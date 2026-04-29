import logging
from typing import List

import anthropic

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise document assistant. Your job is to answer questions based solely on the provided document excerpts.

Rules:
- Answer only from the context provided. Do not use outside knowledge.
- If the context does not contain enough information to answer, say so clearly.
- Be concise and direct. Cite relevant details from the document.
- Do not fabricate or infer beyond what the text states."""


def build_context_block(chunks: List[str]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[Excerpt {i}]\n{chunk}")
    return "\n\n---\n\n".join(parts)


class LLMService:
    """
    Wraps Anthropic Claude for answer generation.

    The prompt is structured as:
        system: role + rules
        user:   context excerpts + question

    Keeping context in the user turn (rather than the system prompt) lets
    Claude cite specific excerpts more reliably.
    """

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def answer(self, question: str, context_chunks: List[str]) -> str:
        """Generate an answer grounded in the provided context chunks."""
        context_block = build_context_block(context_chunks)

        user_message = (
            f"Here are the relevant excerpts from the document:\n\n"
            f"{context_block}\n\n"
            f"---\n\n"
            f"Question: {question}"
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        answer_text = response.content[0].text
        logger.info(f"LLM answered question (input_tokens={response.usage.input_tokens}, output_tokens={response.usage.output_tokens})")
        return answer_text
