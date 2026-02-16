from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


def validate_required_env() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("Missing GOOGLE_API_KEY.")
    if not os.getenv("TAVILY_API_KEY"):
        raise RuntimeError("Missing TAVILY_API_KEY.")


def build_llm() -> ChatGoogleGenerativeAI:
    model = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
    temperature = float(os.getenv("GOOGLE_TEMPERATURE", "0.2"))
    return ChatGoogleGenerativeAI(model=model, temperature=temperature)


def init_runtime() -> None:
    load_dotenv()
    validate_required_env()
