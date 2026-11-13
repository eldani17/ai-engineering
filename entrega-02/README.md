# Pre-entrega 2: Pipeline de procesamiento validado

Pipeline de **extracción de entidades técnicas** hecho con LangChain (LCEL). Recibe un párrafo
de texto sin procesar (una descripción de arquitectura, un log de error, un ticket de incidente)
y devuelve un **objeto Pydantic validado** con las tecnologías mencionadas, el nivel de
criticidad y un resumen técnico.

## Estructura

```
entrega-02/
├── schemas.py          # contrato de datos: NivelCriticidad + ExtraccionTecnica (Pydantic)
├── chain.py            # prompt, cadena LCEL, reintentos, logs y process_text()
├── main.py             # mini-script de prueba asíncrono (4 textos, uno ambiguo)
├── tests/              # tests offline del esquema y de la resiliencia (no llaman a la API)
├── requirements.txt
├── requirements-dev.txt
└── .env.example
```

## Cómo funciona

```
texto ──► ChatPromptTemplate ──► model.with_structured_output(ExtraccionTecnica) ──► verificación ──► objeto validado
          {texto}                └──────────────── .with_retry(): hasta 3 intentos ───────────────┘
          {instrucciones_formato}
```

- **Esquema (`schemas.py`)**: `tecnologias` (lista no vacía, sin duplicados ni strings vacíos),
  `nivel_de_criticidad` (enum `baja` / `media` / `alta`) y `resumen_tecnico` (20 caracteres como
  mínimo). No se aceptan campos extra.
- **Prompt (`chain.py`)**: un `ChatPromptTemplate` con dos variables, `{texto}` e
  `{instrucciones_formato}`. Las instrucciones de formato se generan a partir de las
  descripciones del propio esquema, así no se desincronizan.
- **Cadena LCEL**: `prompt | model.with_structured_output(ExtraccionTecnica)`, con `ChatOpenAI`
  o `ChatAnthropic` según `LLM_PROVIDER` (la misma convención de variables que el cliente del
  Módulo 1).
- **Resiliencia**: `.with_retry()` con backoff exponencial y hasta 3 intentos. Se reintenta
  cuando la salida llega mal formada o incompleta y ante errores transitorios (rate limit,
  timeout, conexión, 5xx). Un 401 o un 400 no se reintentan.
- **`finish_reason`**: antes de aceptar el objeto se revisa el `finish_reason` / `stop_reason`
  de la respuesta cruda. Si el modelo cortó por límite de tokens, la salida se descarta y se
  reintenta.
- **Fallback**: si están cargadas las dos API keys, el otro proveedor queda como respaldo cuando
  el principal agota sus intentos.
- **Logs**: cada llamada al modelo queda registrada con su número de intento, el
  `finish_reason`, los tokens usados, el resultado de la validación y el tipo de error si falla.

## Instalación

Requiere Python 3.11 o superior.

```bash
cd entrega-02
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # y completá la API key
```

Variables del `.env`:

| Variable | Para qué | Default |
|---|---|---|
| `LLM_PROVIDER` | Proveedor principal: `openai` o `anthropic` | `openai` |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | Key del proveedor (la otra es opcional y habilita el fallback) | sin default |
| `OPENAI_MODEL` / `ANTHROPIC_MODEL` | Modelo a usar | `gpt-6.1-sol` / `claude-opus-5-5` |
| `LLM_MAX_TOKENS` | Límite de tokens de salida | `1024` |
| `LLM_TEMPERATURE` | Temperatura (no se envía si no está definida) | sin default |

## Ejecución

```bash
python main.py            # corre los 4 textos en paralelo, con logs
python main.py --quiet    # solo los resultados
```

Desde otro módulo:

```python
import asyncio
from dotenv import load_dotenv
from chain import process_text

load_dotenv()
resultado = asyncio.run(process_text("El servicio de pagos en Kubernetes tira 502..."))
print(resultado.model_dump_json(indent=2))
```

## Ejemplo de salida

Salida real de `python main.py` con `claude-opus-5-5`.

Entrada:

> El servicio de pagos en Kubernetes tira 502 desde el deploy de ayer. Sospechamos del pool de
> conexiones de PostgreSQL; Redis responde bien.

Salida (`resultado.model_dump_json(indent=2)`):

```json
{
  "tecnologias": [
    "Kubernetes",
    "PostgreSQL",
    "Redis"
  ],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "El servicio de pagos desplegado en Kubernetes devuelve errores 502 desde el deploy de ayer. Se sospecha del pool de conexiones de PostgreSQL como causa probable, mientras que Redis funciona correctamente."
}
```

Logs de esa llamada:

```
23:54:33 INFO    pipeline | Procesando texto de 139 caracteres
23:54:33 INFO    pipeline | Intento 1/3: llamando a claude-opus-5-5
23:54:37 INFO    pipeline | Respuesta recibida: finish_reason=end_turn | tokens entrada=1175 salida=140
23:54:37 INFO    pipeline | Validación Pydantic OK (finish_reason=end_turn): 3 tecnologías, criticidad=alta
23:54:37 INFO    pipeline | Extracción completada
```

### Reintentos ante una salida incompleta

Para ver la resiliencia en acción se puede forzar un corte por tokens con
`LLM_MAX_TOKENS=30 python main.py`. El pipeline detecta el `finish_reason`, descarta la salida,
reintenta con backoff y, como el límite sigue siendo el mismo, termina informando el error
(logs reales de una llamada a `process_text` con ese límite):

```
23:55:24 INFO    pipeline | Procesando texto de 68 caracteres
23:55:24 INFO    pipeline | Intento 1/3: llamando a claude-opus-5-5
23:55:26 INFO    pipeline | Respuesta recibida: finish_reason=max_tokens | tokens entrada=1144 salida=30
23:55:26 WARNING pipeline | Salida mal formada o incompleta (SalidaIncompletaError): el modelo cortó la respuesta por límite de tokens (finish_reason=max_tokens)
23:55:27 INFO    pipeline | Intento 2/3: llamando a claude-opus-5-5
23:55:29 INFO    pipeline | Respuesta recibida: finish_reason=max_tokens | tokens entrada=1144 salida=30
23:55:29 WARNING pipeline | Salida mal formada o incompleta (SalidaIncompletaError): el modelo cortó la respuesta por límite de tokens (finish_reason=max_tokens)
23:55:32 INFO    pipeline | Intento 3/3: llamando a claude-opus-5-5
23:55:34 INFO    pipeline | Respuesta recibida: finish_reason=max_tokens | tokens entrada=1144 salida=30
23:55:34 WARNING pipeline | Salida mal formada o incompleta (SalidaIncompletaError): el modelo cortó la respuesta por límite de tokens (finish_reason=max_tokens)
23:55:34 ERROR   pipeline | Sin salida válida después de 3 intentos (SalidaIncompletaError): el modelo cortó la respuesta por límite de tokens (finish_reason=max_tokens)
```

### Prueba de estrés (texto ambiguo)

El cuarto caso de `main.py` no menciona ninguna tecnología:

> Che, ayer anduvo todo medio raro y hoy sigue igual. Alguien que se fije qué pasa porque el
> cliente ya preguntó dos veces.

Como `tecnologias` no puede quedar vacía, había dos desenlaces posibles: que la validación
fallara y se agotaran los reintentos, o que el modelo se las arreglara para cumplir el
contrato. En la corrida real pasó lo segundo, al primer intento:

```json
{
  "tecnologias": [
    "No especificada en el texto"
  ],
  "nivel_de_criticidad": "media",
  "resumen_tecnico": "Se reporta un comportamiento anómalo no especificado que comenzó ayer y persiste hoy, con consultas reiteradas del cliente. El texto no menciona tecnologías, síntomas concretos ni causa probable, por lo que hace falta más información para diagnosticarlo."
}
```

El modelo no inventó una tecnología: completó el campo obligatorio con un marcador y aclaró la
falta de datos en el resumen. Es una limitación conocida del contrato: la validación garantiza
la forma, no que el contenido sea una tecnología real. Si un texto no llegara a producir una
salida válida, el script lo informa como `[SIN SALIDA VÁLIDA]` y los demás se procesan igual,
porque `asyncio.gather` corre con `return_exceptions=True`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

No usan la API: el modelo se reemplaza por respuestas guionadas. Cubren las restricciones del
esquema y que la cadena reintente ante salida mal formada, respuesta cortada por tokens y error
de red, y que no reintente un error de autenticación.

## Decisiones de diseño

- **`with_structured_output` en vez de pedir "respondé en JSON"**: el proveedor garantiza la
  sintaxis y los tipos; Pydantic agrega las reglas de negocio (lista no vacía, enum, largo
  mínimo).
- **`include_raw=True`**: permite mirar el `finish_reason` de la respuesta cruda antes de
  aceptar el objeto, y convierte los errores de parseo en un dato que se puede loguear.
- **Reintentos en un solo lugar**: los clientes se crean con `max_retries=0` para que el único
  que reintenta sea `.with_retry()` y cada intento se vea en los logs.
- **Reintentar solo lo que puede mejorar**: salida inválida y errores transitorios sí; errores de
  autenticación o de request mal armado no, porque reintentar solo gasta tiempo y plata.
