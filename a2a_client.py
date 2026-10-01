import json
import uuid

import httpx

SALARY_AGENT_URL = (
    "http://localhost:8001"
)


async def ask_salary_agent(
    question: str,
) -> dict:
    request = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "messageId": str(uuid.uuid4()),
                "role": "user",
                "parts": [
                    {
                        "kind": "text",
                        "text": question,
                    }
                ],
            }
        },
    }

    async with httpx.AsyncClient(
        timeout=30.0
    ) as client:
        response = await client.post(
            SALARY_AGENT_URL,
            json=request,
        )

        response.raise_for_status()
        payload = response.json()

        print("\n[A2A RESPONSE]")
        print(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            )
        )

        return payload


def extract_salary_answer(
    payload: dict,
) -> str:
    result = payload.get("result", {})

    # Resposta direta como Message
    for part in result.get("parts", []):
        if part.get("kind") == "text":
            content = part.get("text", "")

            try:
                parsed = json.loads(content)
                return parsed.get(
                    "answer",
                    content,
                )
            except Exception:
                return content

    # Caso alguma versão/implementação retorne artifacts
    for artifact in result.get("artifacts", []):
        for part in artifact.get("parts", []):
            if part.get("kind") == "text":
                content = part.get("text", "")

                try:
                    parsed = json.loads(content)
                    return parsed.get(
                        "answer",
                        content,
                    )
                except Exception:
                    return content

    return (
        "Não foi possível obter uma resposta "
        "do agente de Conta Salário."
    )
