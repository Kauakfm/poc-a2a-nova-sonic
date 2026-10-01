# POC - Nova 2 Sonic + A2A + Agente Conta Salário

POC com dois agentes:

1. **Agente de Intenção**
   - Amazon Nova 2 Sonic
   - recebe áudio do microfone
   - conversa com o cliente
   - usa `toolUse` quando precisa consultar Conta Salário

2. **Agente de Conta Salário**
   - exposto via A2A
   - recebe texto
   - faz busca semântica com Amazon Titan Text Embeddings V2
   - consulta um KB local (`kb/conta_salario.txt`)
   - usa Amazon Nova Lite para gerar a resposta
   - devolve texto via A2A

Fluxo:

```text
Microfone
   |
   v
Nova 2 Sonic
   |
   | toolUse
   v
A2A client
   |
   | A2A message/send
   v
Salary Account Agent
   |
   +--> Titan Embeddings V2
   |
   +--> KB local
   |
   +--> Nova Lite
   |
   v
A2A response
   |
   v
toolResult
   |
   v
Nova 2 Sonic
   |
   v
Speaker
```

## Requisitos

- Python 3.12 recomendado
- AWS CLI configurado ou credenciais AWS em variáveis de ambiente
- acesso aos modelos do Bedrock usados no projeto
- microfone e saída de áudio
- PortAudio instalado para `PyAudio`

## Modelos usados

Por padrão:

```text
Nova Sonic:
amazon.nova-2-sonic-v1:0

Embeddings:
amazon.titan-embed-text-v2:0

Resposta do especialista:
amazon.nova-lite-v1:0
```

Podem ser sobrescritos por variáveis de ambiente:

```bash
AWS_REGION=us-east-1
NOVA_SONIC_MODEL_ID=amazon.nova-2-sonic-v1:0
BEDROCK_EMBED_MODEL=amazon.titan-embed-text-v2:0
BEDROCK_ANSWER_MODEL=amazon.nova-lite-v1:0
```

## Instalação

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Depois:

```bash
pip install -r requirements.txt
```

### PyAudio no Windows

Se `pip install pyaudio` falhar, instale o PortAudio/PyAudio de acordo com sua versão de Python.

## Credenciais AWS

Exemplo:

```bash
aws configure
```

Ou configure:

```bash
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_SESSION_TOKEN=...
AWS_REGION=us-east-1
```

A role/usuário precisa poder invocar os modelos usados no Bedrock.

## Rodando

### Terminal 1 - Agente Conta Salário

Na raiz do projeto:

```bash
python -m salary_agent.server
```

Ele inicia em:

```text
http://localhost:8001
```

O KB será vetorizado na inicialização.

### Terminal 2 - Agente de Intenção / Nova Sonic

```bash
python -m intention_agent.sonic
```

Depois fale no microfone algo como:

> Conta salário tem cheque especial?

O fluxo esperado:

```text
Nova Sonic
   |
   | toolUse
   v
consultar_conta_salario
   |
   | A2A
   v
Salary Agent
   |
   | RAG
   v
Nova Lite
   |
   | A2A response
   v
Nova Sonic
   |
   v
resposta em áudio
```

## Observação importante sobre o SDK do Nova Sonic

O SDK Python de bidirectional streaming do Bedrock evolui com frequência.

O arquivo:

```text
intention_agent/sonic.py
```

está estruturado de acordo com o fluxo esperado:

```text
sessionStart
promptStart
contentStart
audioInput
toolUse
toolResult
audioOutput
```

Caso a versão instalada de:

```text
aws-sdk-bedrock-runtime
smithy-aws-core
```

tenha mudado nomes de imports ou tipos, mantenha a arquitetura deste POC e alinhe somente a criação do client/eventos com o sample oficial atual da AWS para Nova 2 Sonic.

Isso não muda a integração A2A.

## Onde está o A2A?

O agente especialista está em:

```text
salary_agent/server.py
```

E o cliente A2A está em:

```text
intention_agent/a2a_client.py
```

A chamada é feita usando:

```text
message/send
```

O payload enviado contém uma mensagem textual.

Isso é intencional: o agente especialista não recebe voz.

## Responsabilidades

### Nova 2 Sonic / Agente de Intenção

Entrada:

```text
áudio
```

Saída ao cliente:

```text
áudio
```

Comunicação com especialistas:

```text
texto / eventos semânticos
```

### Conta Salário

Entrada:

```text
texto
```

Saída:

```text
texto / JSON
```

O especialista não precisa conhecer:

- microfone
- telefone
- URA
- codec
- TTS
- STT
- barge-in

Ele conhece somente o domínio de Conta Salário.
