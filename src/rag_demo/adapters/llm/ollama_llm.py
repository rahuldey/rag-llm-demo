from collections.abc import Iterator

from langchain_ollama import ChatOllama

from rag_demo.domain.ports import LLMPort


class OllamaLLMAdapter(LLMPort):
    def __init__(self, base_url: str, model: str) -> None:
        self._llm = ChatOllama(base_url=base_url, model=model, temperature=0.1)

    def generate(self, prompt: str) -> str:
        response = self._llm.invoke(prompt)
        return str(response.content)

    def generate_stream(self, prompt: str) -> Iterator[str]:
        for chunk in self._llm.stream(prompt):
            if chunk.content:
                yield str(chunk.content)
