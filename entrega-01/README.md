# Pre-entrega 1: Cliente de LLM robusto y asíncrono

Cliente Python asíncrono que habla con **OpenAI** y **Anthropic** detrás de una misma interfaz.
Se elige el proveedor con una variable de entorno, todas las llamadas son `async/await`,
hay streaming de tokens, validación con Pydantic y manejo controlado de errores con reintentos.

## Descripción

La idea es poder escribir una sola vez el código que conversa con un modelo y cambiar de
proveedor sin tocar nada más que el `.env`. Para eso el paquete `llm_client`:

- Define una clase base abstracta (`BaseLLMClient`) con dos operaciones: `generate()` (respuesta completa) y `stream()` (generador asíncrono de fragmentos).
- Implementa esa interfaz para OpenAI (`AsyncOpenAI`, Chat Completions) y Anthropic (`AsyncAnthropic`, Messages API), abstrayendo las diferencias entre SDKs.
- Expone `AsyncLLMManager`, un factory que lee `LLM_PROVIDER` y entrega el cliente correcto, o falla con un mensaje claro si falta la API key.
- Traduce las excepciones de cada SDK a una jerarquía propia y reintenta con backoff exponencial los errores transitorios (rate limit, timeout, conexión, 5xx).

## Arquitectura

```
                 .env  (LLM_PROVIDER, *_API_KEY, *_MODEL)
                   │
                   ▼
   main.py ──► Settings (Pydantic, SecretStr) ──► AsyncLLMManager
                                                      │  factory por proveedor
                                  ┌───────────────────┴───────────────────┐
                                  ▼                                       ▼
                           OpenAIClient                            AnthropicClient
                        (AsyncOpenAI SDK)                        (AsyncAnthropic SDK)
                                  └───────────────┬───────────────────────┘
                                                  ▼
                                           BaseLLMClient
                              generate() / stream() + reintentos + backoff
                                                  │
                                                  ▼
                             ModelResponse (texto, tokens, latencia, error?)
```

Flujo de una llamada:

1. `main.py` carga el `.env`, arma `Settings` y pide un cliente al manager.
2. El cliente recibe una lista de `ChatMessage` validados y arma la request según el SDK.
3. Si el SDK lanza una excepción, el cliente la traduce a una `LLMError` propia (`raise ... from e`).
4. `BaseLLMClient` decide si reintenta (según `retryable` y `max_retries`) y, si se agotan los intentos, devuelve un `ModelResponse` con `error` cargado (en `generate`) o propaga la `LLMError` (en `stream`).

### Estructura del repo

```
entrega-01/
├── llm_client/
│   ├── __init__.py
│   ├── schemas.py          # Role, ChatMessage, ModelConfig, ModelResponse, LLMErrorInfo, Settings
│   ├── exceptions.py       # LLMError y subclases
│   ├── base.py             # BaseLLMClient (ABC) + lógica de reintentos
│   ├── openai_client.py
│   ├── anthropic_client.py
│   └── manager.py          # AsyncLLMManager + factory por proveedor
├── tests/                  # pytest + pytest-asyncio, todo mockeado (sin red)
├── main.py
├── .env.example
├── .gitignore
├── pyproject.toml          # config de pytest
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

### Diferencias entre SDKs que se abstraen

| Tema | OpenAI | Anthropic |
|---|---|---|
| Mensaje `system` | Va dentro de `messages` | Va como parámetro `system=` aparte |
| Texto de la respuesta | `choices[0].message.content` | Bloques de `content` con `type == "text"` |
| Tokens | `usage.prompt_tokens` / `usage.completion_tokens` | `usage.input_tokens` / `usage.output_tokens` |
| Límite de salida | `max_completion_tokens` (opcional) | `max_tokens` (obligatorio) |
| Streaming | `stream=True`, texto en `chunk.choices[0].delta.content` (puede ser `None` o `choices` vacío) | `client.messages.stream(...)` y `text_stream` |
| Fin de respuesta | `finish_reason` | `stop_reason` |

## Requisitos

- Python 3.12
- Una API key de OpenAI, de Anthropic, o las dos.

Dependencias: `openai`, `anthropic`, `pydantic`, `python-dotenv` (y `pytest`, `pytest-asyncio` para los tests).

## Instalación

```bash
cd entrega-01
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # solo runtime
pip install -r requirements-dev.txt    # runtime + tests
```

## Configuración

Copiá el ejemplo y cargá tus keys:

```bash
cp .env.example .env
```

| Variable | Obligatoria | Default | Descripción |
|---|---|---|---|
| `LLM_PROVIDER` | No | `openai` | Proveedor por defecto: `openai` o `anthropic`. |
| `OPENAI_API_KEY` | Si usás OpenAI | – | API key de OpenAI. |
| `ANTHROPIC_API_KEY` | Si usás Anthropic | – | API key de Anthropic. |
| `OPENAI_MODEL` | No | `gpt-6.1-sol` | Modelo de OpenAI (Chat Completions). |
| `ANTHROPIC_MODEL` | No | `claude-opus-5-5` | Modelo de Anthropic (Messages API). |

Los defaults de modelo se verificaron en la documentación oficial de cada proveedor
(catálogo de modelos de OpenAI y referencia del SDK de Anthropic) en noviembre de 2026.
Las keys se guardan como `SecretStr`: nunca se imprimen ni se loguean, y `get_secret_value()`
se llama una sola vez, al construir el cliente del SDK.

> Nota: los modelos de razonamiento actuales (por ejemplo `claude-opus-5-5`) rechazan
> `temperature` y `top_p`. Por eso en `ModelConfig` son opcionales y solo se envían si los seteás.

## Cómo correr `main.py`

```bash
source .venv/bin/activate
python main.py                       # usa LLM_PROVIDER del .env, modo normal + streaming
python main.py --provider anthropic  # pisa el .env
python main.py --mode stream         # solo streaming (o --mode normal)
python main.py --verbose             # muestra logs de reintentos y requests HTTP
```

El script pregunta **"¿Qué es la entropía?"** dos veces: primero en modo normal
(imprime el texto, los tokens y la latencia) y después en streaming (imprime los fragmentos
a medida que llegan y mide el TTFT, *time to first token*, y el tiempo total).
Si algo falla, muestra un mensaje claro y termina con código 1, sin traceback.

## Cómo correr los tests

```bash
source .venv/bin/activate
pytest            # o: pytest -v
```

Los tests no llaman a ninguna API real: los SDKs están mockeados y `asyncio.sleep` del
backoff se reemplaza para que los reintentos no demoren. Cubren:

- Validación de `ModelConfig` (temperatura fuera de 0–2, `max_tokens <= 0`, `top_p` fuera de 0–1, `temperature` + `top_p` juntos) y de `ChatMessage` (contenido vacío, rol inválido).
- `Settings` leyendo del entorno y sin filtrar las keys en `repr`.
- Que el factory cree el cliente correcto según `LLM_PROVIDER` y falle con `LLMConfigError` si falta la key o el proveedor es desconocido.
- Traducción de cada excepción del SDK (OpenAI y Anthropic) a la excepción propia, respetando la herencia, con `__cause__` encadenado.
- Reintentos: rate limit que luego funciona, agotamiento de reintentos, respeto del header `retry-after`, y que un error de autenticación **no** se reintente.
- Streaming: fragmentos esperados (ignorando `delta.content = None` y chunks sin `choices`), reintento antes del primer fragmento y error controlado a mitad del stream.

## Manejo de errores

### Jerarquía propia

```
LLMError
├── LLMConfigError        falta la key / proveedor desconocido      (no reintenta)
├── LLMAuthError          401 / 403                                 (no reintenta)
├── LLMRateLimitError     429, respeta `retry-after`                (reintenta)
├── LLMConnectionError    red caída, DNS, etc.                      (reintenta)
│   └── LLMTimeoutError   la request superó `timeout`               (reintenta)
└── LLMAPIError           cualquier otro status; reintenta solo si es 5xx (incluye 529 de Anthropic)
```

Cada cliente captura las excepciones del SDK y las traduce respetando la herencia: en los dos
SDKs `AuthenticationError` y `RateLimitError` heredan de `APIStatusError`, y `APITimeoutError`
hereda de `APIConnectionError`, así que lo específico se chequea antes que lo genérico.
Siempre se usa `raise ... from e` para conservar la causa original.

### Qué pasa en cada modo

- **`generate()` nunca lanza `LLMError`.** Devuelve un `ModelResponse` con `error` cargado (`LLMErrorInfo`: tipo, mensaje, proveedor, status HTTP, si era reintentable y cuántos intentos se hicieron). `response.success` es `False`. Así el loop principal no se cae nunca por un problema de red o de cuota.
- **`stream()` lanza la excepción propia.** Elegimos esta opción porque un generador no puede "devolver" un objeto de error sin ensuciar el tipo de lo que emite. El consumidor (`main.py`) envuelve el `async for` en un `try/except LLMError`. Los reintentos solo aplican **antes del primer fragmento**: una vez que empezó a salir texto, no se puede rebobinar sin duplicar salida, así que el error se propaga.
- **`AsyncLLMManager.create_client()` lanza `LLMConfigError`** si falta la key o el proveedor no existe; `main.py` también lo captura y lo muestra sin traceback.

### Reintentos

La lógica está explícita en `BaseLLMClient._run_with_retries` (los SDKs se crean con
`max_retries=0` para que no haya doble reintento):

- Se reintenta solo si `error.retryable` es `True`: rate limit, timeout, conexión y 5xx.
- Cantidad de reintentos: `ModelConfig.max_retries` (default 3).
- Espera: backoff exponencial `min(0.5 · 2^intento, 8)` segundos, más un jitter aleatorio de hasta 0,25 s. Si el proveedor mandó `retry-after`, se espera al menos eso.
- Errores de autenticación y 4xx (salvo 429) **no** se reintentan.

## Ejemplo de salida

Salida real obtenida con `claude-opus-5-5` (recortada):

```
=== Modo normal (anthropic / claude-opus-5-5) ===
La entropía es una magnitud física que mide, a grandes rasgos, cuántas formas distintas
tiene un sistema de estar organizado por dentro sin que notemos la diferencia desde afuera ...

-- tokens entrada: 52 | tokens salida: 757 | latencia: 9334 ms | finish_reason: end_turn

=== Modo streaming (anthropic / claude-opus-5-5) ===
La entropía es una magnitud física que mide, en términos simples, cuántas formas distintas
tiene un sistema de estar organizado por dentro ...

-- TTFT: 1201 ms | total: 7654 ms | fragmentos: 130
```

Con una key inválida (error controlado, sin traceback, exit code 1):

```
=== Modo normal (openai / gpt-6.1-sol) ===

[ERROR] LLMAuthError: API key de OpenAI inválida o ausente
        intentos: 1 | reintentable: False

=== Modo streaming (openai / gpt-6.1-sol) ===

[ERROR] LLMAuthError: API key de OpenAI inválida o ausente
        intentos: 1 | reintentable: False
```

Sin key cargada:

```
[ERROR] LLMConfigError: falta la variable ANTHROPIC_API_KEY para usar el proveedor 'anthropic'. Cargala en el .env o en el entorno.
```

## Uso como librería

```python
import asyncio
from dotenv import load_dotenv
from llm_client import AsyncLLMManager, ChatMessage, LLMError, Role, Settings

async def main() -> None:
    load_dotenv()
    async with AsyncLLMManager(Settings.from_env()) as manager:
        client = manager.get_client()  # o manager.get_client("anthropic")
        mensajes = [ChatMessage(role=Role.USER, content="¿Qué es la entropía?")]

        respuesta = await client.generate(mensajes)
        print(respuesta.text if respuesta.success else respuesta.error)

        try:
            async for fragmento in client.stream(mensajes):
                print(fragmento, end="", flush=True)
        except LLMError as error:
            print(f"\nfalló el streaming: {error}")

asyncio.run(main())
```
