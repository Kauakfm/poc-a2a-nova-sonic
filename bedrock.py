import json
import os
import boto3

REGION = os.getenv("AWS_REGION", "us-east-1")

EMBED_MODEL = os.getenv(
    "BEDROCK_EMBED_MODEL",
    "amazon.titan-embed-text-v2:0",
)

ANSWER_MODEL = os.getenv(
    "BEDROCK_ANSWER_MODEL",
    "amazon.nova-lite-v1:0",
)

bedrock = boto3.client(
    "bedrock-runtime",
    region_name=REGION,
)


def embed(text: str) -> list[float]:
    response = bedrock.invoke_model(
        modelId=EMBED_MODEL,
        body=json.dumps(
            {
                "inputText": text,
                "dimensions": 1024,
                "normalize": True,
            }
        ),
        contentType="application/json",
        accept="application/json",
    )

    body = json.loads(response["body"].read())
    return body["embedding"]


def answer_with_context(question: str, context: str) -> str:
    prompt = f"""
Você é um agente especialista em Conta Salário.

Responda exclusivamente com base no CONTEXTO abaixo.
Se não houver informação suficiente, diga:
"Não encontrei essa informação na base de conhecimento de Conta Salário."

Não invente informações.

CONTEXTO:
{context}

PERGUNTA:
{question}

Responda de maneira objetiva, clara e apropriada para um atendimento bancário.
""".strip()

    response = bedrock.converse(
        modelId=ANSWER_MODEL,
        messages=[
            {
                "role": "user",
                "content": [{"text": prompt}],
            }
        ],
        inferenceConfig={
            "temperature": 0.1,
            "maxTokens": 300,
        },
    )

    return response["output"]["message"]["content"][0]["text"]
