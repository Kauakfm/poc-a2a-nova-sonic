import asyncio
import base64
import json
import os
import uuid

import pyaudio

from aws_sdk_bedrock_runtime.client import (
    BedrockRuntimeClient,
    InvokeModelWithBidirectionalStreamOperationInput,
)
from aws_sdk_bedrock_runtime.models import (
    InvokeModelWithBidirectionalStreamInputChunk,
    BidirectionalInputPayloadPart,
)
from aws_sdk_bedrock_runtime.config import Config

# OBS:
# Dependendo da versão instalada do SDK preview/GA de streaming,
# imports de credenciais/auth podem mudar.
# O README explica como alinhar esta parte ao sample oficial atual
# da AWS caso sua versão do SDK seja diferente.
from smithy_aws_core.identity.environment import (
    EnvironmentCredentialsResolver,
)
from smithy_aws_core.auth.sigv4 import (
    SigV4AuthScheme,
)
from smithy_aws_core.auth.resolver import (
    HTTPAuthSchemeResolver,
)

from intention_agent.a2a_client import (
    ask_salary_agent,
    extract_salary_answer,
)


REGION = os.getenv("AWS_REGION", "us-east-1")
MODEL_ID = os.getenv(
    "NOVA_SONIC_MODEL_ID",
    "amazon.nova-2-sonic-v1:0",
)

INPUT_RATE = 16000
OUTPUT_RATE = 24000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 1024


SYSTEM_PROMPT = """
Você é o agente de intenção de uma URA bancária.

Você conversa diretamente com o cliente por voz.

Sua responsabilidade é entender a solicitação do cliente
e decidir quando consultar agentes especialistas.

Atualmente existe apenas um agente especialista:
Conta Salário.

Sempre que a pergunta envolver Conta Salário,
portabilidade salarial, recebimento de salário,
características de conta salário ou dúvidas relacionadas,
use a ferramenta consultar_conta_salario.

Não invente informações de Conta Salário.

Quando receber a resposta da ferramenta,
use o conteúdo retornado para responder ao cliente
de maneira natural, curta e adequada para conversa telefônica.

Não mencione ferramentas, agentes, A2A,
arquitetura ou detalhes técnicos para o cliente.
""".strip()


class SonicIntentionAgent:
    def __init__(self):
        endpoint = (
            f"https://bedrock-runtime."
            f"{REGION}.amazonaws.com"
        )

        config = Config(
            endpoint_uri=endpoint,
            region=REGION,
            aws_credentials_identity_resolver=(
                EnvironmentCredentialsResolver()
            ),
            http_auth_scheme_resolver=(
                HTTPAuthSchemeResolver()
            ),
            http_auth_schemes={
                "aws.auth#sigv4":
                    SigV4AuthScheme()
            },
        )

        self.client = BedrockRuntimeClient(
            config=config
        )

        self.stream = None
        self.prompt_name = str(uuid.uuid4())
        self.audio = pyaudio.PyAudio()
        self.output_stream = None
        self._tool_buffers: dict[str, str] = {}

    async def send_event(
        self,
        payload: dict,
    ) -> None:
        raw = json.dumps(
            payload
        ).encode("utf-8")

        chunk = (
            InvokeModelWithBidirectionalStreamInputChunk(
                value=BidirectionalInputPayloadPart(
                    bytes_=raw
                )
            )
        )

        await self.stream.input_stream.send(
            chunk
        )

    def prompt_start_event(self) -> dict:
        return {
            "event": {
                "promptStart": {
                    "promptName":
                        self.prompt_name,

                    "textOutputConfiguration": {
                        "mediaType":
                            "text/plain"
                    },

                    "audioOutputConfiguration": {
                        "mediaType":
                            "audio/lpcm",
                        "sampleRateHertz":
                            OUTPUT_RATE,
                        "sampleSizeBits":
                            16,
                        "channelCount":
                            1,
                        "voiceId":
                            "matthew",
                        "encoding":
                            "base64",
                        "audioType":
                            "SPEECH",
                    },

                    "toolUseOutputConfiguration": {
                        "mediaType":
                            "application/json"
                    },

                    "toolConfiguration": {
                        "tools": [
                            {
                                "toolSpec": {
                                    "name":
                                        "consultar_conta_salario",
                                    "description": (
                                        "Consulta o agente "
                                        "especialista de Conta "
                                        "Salário quando o cliente "
                                        "tiver uma dúvida sobre "
                                        "Conta Salário."
                                    ),
                                    "inputSchema": {
                                        "json": json.dumps(
                                            {
                                                "type":
                                                    "object",
                                                "properties":
                                                    {
                                                        "question":
                                                            {
                                                                "type":
                                                                    "string",
                                                                "description":
                                                                    (
                                                                        "Pergunta do cliente "
                                                                        "sobre Conta Salário."
                                                                    ),
                                                            }
                                                    },
                                                "required":
                                                    [
                                                        "question"
                                                    ],
                                            }
                                        )
                                    },
                                }
                            }
                        ]
                    },
                }
            }
        }

    async def start(self) -> None:
        self.stream = await (
            self.client
            .invoke_model_with_bidirectional_stream(
                InvokeModelWithBidirectionalStreamOperationInput(
                    model_id=MODEL_ID
                )
            )
        )

        await self.send_event(
            {
                "event": {
                    "sessionStart": {
                        "inferenceConfiguration": {
                            "maxTokens": 1024,
                            "temperature": 0.3,
                            "topP": 0.9,
                        }
                    }
                }
            }
        )

        await self.send_event(
            self.prompt_start_event()
        )

        await self.send_text(
            role="SYSTEM",
            text=SYSTEM_PROMPT,
        )

    async def send_text(
        self,
        role: str,
        text: str,
    ) -> None:
        content_name = str(uuid.uuid4())

        await self.send_event(
            {
                "event": {
                    "contentStart": {
                        "promptName":
                            self.prompt_name,
                        "contentName":
                            content_name,
                        "type":
                            "TEXT",
                        "interactive":
                            False,
                        "role":
                            role,
                    }
                }
            }
        )

        await self.send_event(
            {
                "event": {
                    "textInput": {
                        "promptName":
                            self.prompt_name,
                        "contentName":
                            content_name,
                        "content":
                            text,
                    }
                }
            }
        )

        await self.send_event(
            {
                "event": {
                    "contentEnd": {
                        "promptName":
                            self.prompt_name,
                        "contentName":
                            content_name,
                    }
                }
            }
        )

    async def send_tool_result(
        self,
        tool_use_id: str,
        result: dict,
    ) -> None:
        content_name = str(uuid.uuid4())

        await self.send_event(
            {
                "event": {
                    "contentStart": {
                        "promptName":
                            self.prompt_name,
                        "contentName":
                            content_name,
                        "type":
                            "TOOL",
                        "interactive":
                            False,
                        "role":
                            "TOOL",
                        "toolResultInputConfiguration":
                            {
                                "toolUseId":
                                    tool_use_id,
                                "type":
                                    "TEXT",
                                "textInputConfiguration":
                                    {
                                        "mediaType":
                                            "text/plain"
                                    },
                            },
                    }
                }
            }
        )

        await self.send_event(
            {
                "event": {
                    "toolResult": {
                        "promptName":
                            self.prompt_name,
                        "contentName":
                            content_name,
                        "content":
                            json.dumps(
                                result,
                                ensure_ascii=False,
                            ),
                    }
                }
            }
        )

        await self.send_event(
            {
                "event": {
                    "contentEnd": {
                        "promptName":
                            self.prompt_name,
                        "contentName":
                            content_name,
                    }
                }
            }
        )

    async def handle_tool_use(
        self,
        event: dict,
    ) -> None:
        tool_use = event["toolUse"]

        tool_name = tool_use["toolName"]
        tool_use_id = tool_use["toolUseId"]

        content = tool_use.get(
            "content",
            "{}",
        )

        if isinstance(content, str):
            input_data = json.loads(content)
        else:
            input_data = content

        print("\n[SONIC TOOL USE]")
        print("tool:", tool_name)
        print("input:", input_data)

        if (
            tool_name
            == "consultar_conta_salario"
        ):
            question = input_data["question"]

            a2a_response = (
                await ask_salary_agent(
                    question
                )
            )

            answer = extract_salary_answer(
                a2a_response
            )

            await self.send_tool_result(
                tool_use_id,
                {
                    "answer": answer,
                },
            )

    async def play_audio(
        self,
        audio_bytes: bytes,
    ) -> None:
        if self.output_stream is None:
            self.output_stream = (
                self.audio.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=OUTPUT_RATE,
                    output=True,
                )
            )

        self.output_stream.write(
            audio_bytes
        )

    async def receive_events(
        self,
    ) -> None:
        async for output in (
            self.stream.output_stream
        ):
            payload = json.loads(
                output.value.bytes.decode(
                    "utf-8"
                )
            )

            event = payload.get(
                "event",
                {},
            )

            if "toolUse" in event:
                await self.handle_tool_use(
                    event
                )

            elif "textOutput" in event:
                content = (
                    event["textOutput"]
                    .get("content")
                )

                if content:
                    print(
                        "[SONIC TEXT]",
                        content,
                    )

            elif "audioOutput" in event:
                content = (
                    event["audioOutput"]
                    .get("content")
                )

                if content:
                    audio_data = (
                        base64.b64decode(
                            content
                        )
                    )

                    await self.play_audio(
                        audio_data
                    )

    async def microphone_loop(
        self,
    ) -> None:
        input_stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=INPUT_RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )

        audio_content_name = (
            str(uuid.uuid4())
        )

        await self.send_event(
            {
                "event": {
                    "contentStart": {
                        "promptName":
                            self.prompt_name,
                        "contentName":
                            audio_content_name,
                        "type":
                            "AUDIO",
                        "interactive":
                            True,
                        "role":
                            "USER",
                        "audioInputConfiguration":
                            {
                                "mediaType":
                                    "audio/lpcm",
                                "sampleRateHertz":
                                    INPUT_RATE,
                                "sampleSizeBits":
                                    16,
                                "channelCount":
                                    1,
                                "audioType":
                                    "SPEECH",
                                "encoding":
                                    "base64",
                            },
                    }
                }
            }
        )

        print(
            "\n🎤 Pode falar. Ctrl+C para sair.\n"
        )

        try:
            while True:
                data = input_stream.read(
                    CHUNK,
                    exception_on_overflow=False,
                )

                encoded = (
                    base64.b64encode(
                        data
                    ).decode("utf-8")
                )

                await self.send_event(
                    {
                        "event": {
                            "audioInput": {
                                "promptName":
                                    self.prompt_name,
                                "contentName":
                                    audio_content_name,
                                "content":
                                    encoded,
                            }
                        }
                    }
                )

                await asyncio.sleep(0)

        finally:
            await self.send_event(
                {
                    "event": {
                        "contentEnd": {
                            "promptName":
                                self.prompt_name,
                            "contentName":
                                audio_content_name,
                        }
                    }
                }
            )

            input_stream.stop_stream()
            input_stream.close()

    async def run(self) -> None:
        await self.start()

        try:
            await asyncio.gather(
                self.microphone_loop(),
                self.receive_events(),
            )
        finally:
            if self.output_stream:
                self.output_stream.stop_stream()
                self.output_stream.close()

            self.audio.terminate()


async def main():
    agent = SonicIntentionAgent()
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
