from pathlib import Path

import numpy as np

from shared.bedrock import embed, answer_with_context

KB_PATH = (
    Path(__file__).parent.parent
    / "kb"
    / "conta_salario.txt"
)


class SalaryRAG:
    def __init__(self):
        self.chunks: list[str] = []
        self.vectors: list[np.ndarray] = []

    def load(self) -> None:
        content = KB_PATH.read_text(encoding="utf-8")

        self.chunks = [
            chunk.strip()
            for chunk in content.split("\n\n")
            if chunk.strip()
        ]

        self.vectors = []

        for chunk in self.chunks:
            self.vectors.append(
                np.array(embed(chunk), dtype=np.float32)
            )

        print(
            f"[SALARY] {len(self.chunks)} chunks indexados."
        )

    def search(
        self,
        question: str,
        k: int = 3,
    ) -> list[dict]:
        query_vector = np.array(
            embed(question),
            dtype=np.float32,
        )

        ranking = []

        for index, vector in enumerate(self.vectors):
            score = float(
                np.dot(query_vector, vector)
            )
            ranking.append((score, index))

        ranking.sort(reverse=True)

        return [
            {
                "score": score,
                "text": self.chunks[index],
            }
            for score, index in ranking[:k]
        ]

    def ask(self, question: str) -> dict:
        documents = self.search(question)

        context = "\n\n".join(
            doc["text"]
            for doc in documents
        )

        answer = answer_with_context(
            question,
            context,
        )

        return {
            "answer": answer,
            "documents": documents,
        }


salary_rag = SalaryRAG()
