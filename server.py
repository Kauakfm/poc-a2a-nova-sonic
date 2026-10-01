import json

import uvicorn

from a2a.server.agent_execution import (
    AgentExecutor,
    RequestContext,
)
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.request_handlers import (
    DefaultRequestHandler,
)
from a2a.server.apps import (
    A2AStarletteApplication,
)
from a2a.types import (
    AgentCard,
    AgentSkill,
    AgentCapabilities,
)
from a2a.utils import new_agent_text_message

from salary_agent.rag import salary_rag


class SalaryAgentExecutor(AgentExecutor):
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        question = context.get_user_input()

        print(
            "\n[SALARY AGENT] "
            f"Pergunta recebida via A2A: {question}"
        )

        result = salary_rag.ask(question)

        print(
            "[SALARY AGENT] resposta:",
            result["answer"],
        )

        await event_queue.enqueue_event(
            new_agent_text_message(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
            )
        )

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        raise NotImplementedError(
            "Cancel não implementado neste POC."
        )


skill = AgentSkill(
    id="salary-account-qa",
    name="Conta Salário",
    description=(
        "Responde dúvidas sobre Conta Salário."
    ),
    tags=[
        "conta-salario",
        "salario",
        "portabilidade",
    ],
    examples=[
        "Conta salário tem cheque especial?",
        "Posso fazer portabilidade?",
        "Como funciona conta salário?",
    ],
)

card = AgentCard(
    name="Salary Account Agent",
    description=(
        "Agente especialista em Conta Salário."
    ),
    url="http://localhost:8001",
    version="1.0.0",
    capabilities=AgentCapabilities(
        streaming=False,
    ),
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain"],
    skills=[skill],
)


def build_app():
    executor = SalaryAgentExecutor()

    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    return A2AStarletteApplication(
        agent_card=card,
        http_handler=handler,
    ).build()


if __name__ == "__main__":
    salary_rag.load()

    uvicorn.run(
        build_app(),
        host="0.0.0.0",
        port=8001,
    )
