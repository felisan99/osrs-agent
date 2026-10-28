# OSRS Agent — Especificación técnica v0

> Especificación base del proyecto. Donde dice **[VERIFICAR]**, el dato se confirma contra la documentación oficial vigente antes de implementarlo (resultado en [`docs/verification.md`](docs/verification.md)).

## 1. Objetivo

Agente de dominio cerrado para **Old School RuneScape (OSRS)**, expuesto a través de un **plugin de RuneLite** (cliente delgado en Java) que habla con un **backend propio** (Python + LangGraph).

Capacidades:

- Responder preguntas sobre el juego con **Corrective RAG (CRAG)** sobre un dump del Wiki de OSRS.
- Workflows especializados: **quests**, **moneymaking** (con precios vivos del Grand Exchange) y **flipping**.
- Usar el **estado del personaje** (skills, quests, inventario, banco, GP, tipo de cuenta) para personalizar respuestas.
- Conversación **multi-turno** con memoria de hilo.

Modelo de ejecución: **BYOK** sobre OpenRouter. El usuario aporta su propia API key; el costo de inferencia no lo paga el desarrollador.

### Principios de diseño

1. **Los números los calcula el código, no el LLM.** Márgenes, ROI, profit/hora, impuestos y rankings son funciones deterministas y testeables.
2. **Jev decide, el LLM razona.** Jev (modelo de decisión de TypeSafe) clasifica y evalúa con probabilidades tipadas. El LLM genera texto y extrae argumentos de tools.
3. **Sin evidencia, no se inventa.** Si el CRAG no encuentra contexto suficiente tras la corrección, el agente lo dice.
4. **Puramente informativo.** El agente nunca automatiza acciones en el juego (reglas de Jagex para plugins).
5. **La key del usuario es un secreto.** Pass-through por request: nunca se persiste ni se loguea.

### Fuera de alcance del v0

Overlays o resaltado en el juego, memoria de largo plazo entre sesiones, seguimiento de offers del GE, actualización periódica del dump, streaming de tokens, publicación en el Plugin Hub, perfiles persistidos de jugador.

## 2. Arquitectura

```
[RuneLite plugin (Java)]
   │  POST /v1/chat  {thread_id, message, player_state}
   │  header: X-OpenRouter-Key (pass-through)
   ▼
[FastAPI backend] ── LangGraph ──┬─ Jev (OpenRouter Decisions API): routing + grading
                                 ├─ LLM barato (OpenRouter chat completions): razonamiento / generación / tool args
                                 ├─ Retrieval híbrido (Supabase Postgres: pgvector + FTS)
                                 ├─ Tools deterministas (API de precios del Wiki)
                                 └─ Checkpointer (Postgres) → memoria de hilo
```

### Modelos y APIs

| Rol | Servicio | Notas |
|---|---|---|
| Router + evaluador CRAG | `typesafe/jev-1.13` vía OpenRouter **Decisions API** (`POST https://openrouter.ai/api/alpha/decisions`) | Devuelve probabilidades tipadas, no texto. Primitivas: **Choice**, **Noul**, **Score**. Es un endpoint **alpha**; no es compatible con el chat de OpenAI. Hay SDK de Python. **[VERIFICAR]** la forma exacta de request/response en la documentación (Decisions API reference y la página de "State" de TypeSafe para referenciar campos anidados). |
| Razonamiento / generación / tool calling | Modelo barato de OpenRouter (p. ej. DeepSeek) vía `POST https://openrouter.ai/api/v1/chat/completions` | El slug del modelo es configurable (`REASONING_MODEL`). **[VERIFICAR]** el slug vigente y que el provider soporte function calling. |
| Embeddings | Modelo local multilingüe (default `intfloat/multilingual-e5-small`, 384 dim) vía `sentence-transformers` | Consultas en español contra un Wiki en inglés. Local para no gastar la key BYOK. Respetar prefijos `query: ` / `passage: ` de E5. Configurable. |
| Precios GE | `https://prices.runescape.wiki/api/v2/osrs` | Ver §6. |
| Base de datos | Supabase (Postgres + pgvector) | Local con Supabase CLI para el v0. |

> Contexto de Jev: la documentación dice 32K tokens y la página del modelo dice 64K. **Asumí 32K** (incluye `state` + preguntas) para dimensionar los batches, y hacé el tamaño de batch configurable.

## 3. Estructura del repo

```
osrs-agent/
├── SPEC.md
├── backend/
│   ├── pyproject.toml
│   ├── .env.example
│   ├── app/
│   │   ├── main.py                 # FastAPI
│   │   ├── config.py               # settings (pydantic-settings)
│   │   ├── api/schemas.py          # contratos request/response
│   │   ├── clients/
│   │   │   ├── openrouter.py       # chat completions con key por request
│   │   │   ├── jev.py              # adapter de Decisions API (aislar aquí el formato)
│   │   │   ├── prices.py           # API de precios del Wiki (User-Agent descriptivo)
│   │   │   └── wiki.py             # API MediaWiki (search tool en vivo)
│   │   ├── graph/
│   │   │   ├── state.py            # AgentState (TypedDict/Pydantic)
│   │   │   ├── build.py            # construcción del grafo + checkpointer
│   │   │   └── nodes/              # un archivo por nodo o subgrafo
│   │   ├── rag/
│   │   │   ├── embed.py
│   │   │   ├── retrieve.py         # híbrido + RRF
│   │   │   └── grade.py            # evaluación con Jev
│   │   ├── tools/
│   │   │   ├── items.py            # resolución nombre→id (mapping + fuzzy)
│   │   │   ├── prices.py           # get_item_price, get_price_history
│   │   │   ├── flips.py            # calc_flip, rank_flips
│   │   │   └── moneymakers.py      # rank_moneymakers
│   │   └── ingest/
│   │       ├── dump.py             # descarga/lectura del dump
│   │       ├── parse.py            # wikitext → estructura
│   │       ├── chunk.py
│   │       └── load.py             # embeddings + carga a Postgres
│   ├── eval/
│   │   ├── questions.yaml          # 30–50 preguntas con respuesta esperada
│   │   └── run_eval.py
│   └── tests/
├── plugin/                         # plugin de RuneLite (Java, Gradle)
│   └── src/main/java/.../
├── supabase/
│   └── migrations/
└── docs/
```

## 4. Contrato HTTP (plugin ↔ backend)

### `POST /v1/chat`

Headers:
- `Content-Type: application/json`
- `X-OpenRouter-Key: <key del usuario>` — **nunca** persistir ni loguear. Redactar en logs de acceso y de errores.

Body:
```json
{
  "thread_id": "uuid-generado-por-sesión-del-plugin",
  "message": "¿Qué moneymaker me conviene con mis stats?",
  "player_state": { "...ver §5..." }
}
```

Response:
```json
{
  "answer": "texto en markdown simple",
  "intent": "moneymaking",
  "sources": [{"title": "Money making guide/...", "url": "https://oldschool.runescape.wiki/..."}],
  "debug": {"route_probs": {}, "retrieval_iterations": 1, "tools_called": []}
}
```

- `debug` solo se devuelve si `DEBUG_RESPONSES=true`.
- Errores tipados: `401 invalid_key`, `402 insufficient_credits`, `429 rate_limited`, `502 upstream_error`. Mapear los errores de OpenRouter a estos códigos.
- `GET /healthz` para el plugin.

## 5. Estado del jugador (`player_state`)

Snapshot compacto, enviado en cada turno (el backend es stateless respecto al jugador):

```json
{
  "account_type": "main | ironman | hardcore_ironman | ultimate_ironman | group_ironman | unknown",
  "members": true,
  "skills": {"attack": 80, "...": 0},
  "quests": {"Dragon Slayer II": "FINISHED", "Monkey Madness II": "NOT_STARTED"},
  "gp": 1250000,
  "inventory": [{"id": 4151, "qty": 1}],
  "bank": {"captured_at": "ISO-8601 | null", "items": [{"id": 995, "qty": 1000000}]},
  "captured_at": "ISO-8601"
}
```

Reglas:
- El banco solo es legible cuando el jugador lo abre. Se cachea en el plugin con `captured_at`; el backend debe tratar un banco viejo o nulo como **dato potencialmente desactualizado** y decirlo cuando importe.
- Estados de quest: usar los del enum de RuneLite (`NOT_STARTED`, `IN_PROGRESS`, `FINISHED`).
- `members`: **[VERIFICAR]** cómo detectarlo desde el cliente; si no es confiable, `null` y preguntar al usuario cuando afecte la respuesta.
- Los `item id` se resuelven a nombres en el backend con `/mapping`.

## 6. Tools de precios (deterministas)

Base: `https://prices.runescape.wiki/api/v2/osrs`

| Ruta | Uso |
|---|---|
| `/latest` | Último `high`/`low` y timestamps de todos los ítems. `?id=` solo para consultas puntuales; **nunca** iterar ítem por ítem. |
| `/mapping` | Metadatos: `id`, `name`, `members`, `limit` (buy limit), `highalch`, `lowalch`, `value`. Cachear (cambia poco). |
| `/5m`, `/1h` | Promedios `avgHighPrice`/`avgLowPrice` y volúmenes `highPriceVolume`/`lowPriceVolume`. |
| `/timeseries?id=&lookback=` | Histórico. `lookback` ∈ `6h, 24h, 7d, 30d, 6m, 1y`. |

Requisitos obligatorios:
- **User-Agent descriptivo** en todas las requests (p. ej. `osrs-agent-dev - contacto@...`). Los user-agents genéricos (`python-requests`, `curl`, `Python-urllib`, etc.) están bloqueados.
- Los precios pueden superar int32: usar enteros de Python y no asumir 32 bits.
- Cachear `/latest` y `/5m` con TTL corto (30–60 s) y `/mapping` con TTL largo. Agrupar requests.
- Semántica: `high` = precio al que se compra al instante, `low` = precio al que se vende al instante. Un flip típico compra cerca de `low` y vende cerca de `high`.

### Impuesto del GE

Parametrizado en `config.py`: `GE_TAX_RATE` (default `0.02`), `GE_TAX_CAP`, y la lista de ítems exentos. **[VERIFICAR]** los valores vigentes en el Wiki (página "Grand Exchange", sección de impuesto) antes de fijar defaults. El impuesto se aplica al lado de venta. No hardcodear en la lógica.

### Tools expuestas al LLM (function calling)

Todas devuelven JSON estructurado; el LLM no calcula.

- `get_item_price(item: str)` → resuelve nombre (fuzzy con `rapidfuzz` sobre `/mapping`; si es ambiguo devuelve candidatos) y retorna high/low, timestamps, volumen reciente y buy limit.
- `get_price_history(item: str, lookback: str)` → serie resumida (min/max/mediana/tendencia) más puntos downsampleados.
- `calc_flip(item: str, qty: int | null)` → margen post-impuesto, ROI, límite por 4 h, ganancia máxima por ventana de límite, volumen para estimar liquidez.
- `rank_flips(max_gp: int, members: bool, min_volume: int, top_n: int)` → ranking sobre todo `/latest` + `/1h` + `/mapping`, respetando el GP disponible.
- `rank_moneymakers(player_state)` → ver §8.

## 7. Grafo (LangGraph)

### Estado (`AgentState`)

`messages`, `player_state`, `intent`, `route_probs`, `query` (reescrita), `candidates`, `graded` (pasajes con probabilidad), `sufficient: bool | None`, `retrieval_iterations`, `tool_results`, `answer`, `sources`.

### Nodos

1. **`classify_intent`** — Jev **Choice** sobre {`quest`, `moneymaking`, `flipping`, `price_lookup`, `wiki_general`, `out_of_domain`}. El `state` incluye el último mensaje, un resumen corto del historial y un resumen del `player_state`. Si la confianza es baja, caer a `wiki_general`. `out_of_domain` responde con un rechazo breve y amable sin llamar al LLM.
2. **Subgrafo CRAG** (`retrieve → grade → [rewrite → search_live → grade]* → generate`):
   - `retrieve`: híbrido (vector + FTS de Postgres) con **Reciprocal Rank Fusion**, top-k configurable (default 20).
   - `grade`: **una sola** llamada batch a Jev con todos los candidatos. Un **Noul por pasaje** ("¿ayuda a responder la consulta?") y un **Noul de suficiencia** ("¿el conjunto retenido permite responder?"). Umbrales en config: `REL_THRESHOLD=0.70`, `SUFFICIENCY_THRESHOLD=0.50` (puntos de partida; se ajustan con la eval). Pasajes bajo el umbral se descartan.
   - `rewrite`: el LLM reescribe la query (en inglés, términos del juego) si no hay suficiencia.
   - `search_live`: consulta a la API de MediaWiki (`action=query&list=search` + extracts) para traer páginas candidatas; se re-evalúan con `grade`.
   - Tope: `MAX_CRAG_ITERATIONS=2`. Si sigue sin evidencia suficiente → responder que no se encontró información confiable, sin inventar.
   - `generate`: LLM con el contexto filtrado, el `player_state` resumido y el historial. Debe citar las páginas usadas.
3. **`quest_workflow`** — CRAG sobre páginas de la quest + comparación determinista requisitos vs `player_state` (skills, quests previas, ítems). Salida: qué falta, orden sugerido, ítems necesarios, pasos.
4. **`moneymaking_workflow`** — `rank_moneymakers` + CRAG para explicar/detallar la guía elegida. Ver §8.
5. **`flipping_workflow`** — `rank_flips` / `calc_flip` con tool calling; el LLM explica y advierte riesgos (volumen bajo, buy limit, precios desactualizados).
6. **`price_lookup`** — tool calling directo (`get_item_price`, `get_price_history`).
7. **`out_of_domain`** — rechazo corto.

Memoria: checkpointer de Postgres de LangGraph (`langgraph-checkpoint-postgres`) con `thread_id` del request. **[VERIFICAR]** la API vigente de `PostgresSaver`/`AsyncPostgresSaver` y su `setup()`.

### Reglas para los prompts del LLM

- Responder en el idioma del usuario (default español); los nombres de ítems, quests y NPCs se mantienen en inglés tal como aparecen en el juego.
- Prohibido dar números de precios o ganancias que no provengan de una tool.
- Indicar siempre el timestamp de los precios usados.
- Si falta un dato del jugador necesario (p. ej. banco no capturado), decirlo en lugar de suponerlo.
- No sugerir ni describir automatización, bots ni macros.

## 8. Moneymaking

### Ingesta estructurada

Las guías de moneymaking del Wiki contienen datos semi-estructurados (actividad, requisitos, ítems de entrada/salida, tasas por hora) y **precios embebidos que quedan viejos**. Para el v0:

1. **[VERIFICAR]** inspeccionar el wikitext real de varias páginas bajo `Money making guide/` e identificar las plantillas y parámetros que contienen requisitos y listas de ítems.
2. Si el formato es consistente → parser determinista con `mwparserfromhell` a la tabla `moneymaking_guides`.
3. Si no → **una sola pasada de extracción con LLM en la ingesta** (nunca en tiempo de consulta), validando con Pydantic y guardando el resultado.

Campos mínimos: `page_title`, `activity`, `category`, `skill_requirements` (dict skill→nivel), `quest_requirements` (lista), `items_in` (lista id/nombre + cantidad por hora), `items_out` (ídem), `members_only`, `ironman_friendly` (si se puede inferir, si no `null`), `source_hourly_profit` (el número histórico del Wiki, solo como referencia).

### `rank_moneymakers(player_state)`

1. Filtrar guías cuyos requisitos de skills y quests cumple el jugador (`members` y `account_type` incluidos). Reportar también "casi elegibles" (a pocos niveles) como categoría aparte.
2. Recalcular profit/hora con precios vivos: `Σ(items_out × precio de venta neto de impuesto) − Σ(items_in × precio de compra)`. Usar `avgHighPrice`/`avgLowPrice` de `/1h` o `/latest` según liquidez, y excluir ítems con volumen insuficiente.
3. Para ironmen, no usar precios de GE como fuente de ingreso; marcar como no aplicable o usar una heurística explícita. **[VERIFICAR]** con el usuario antes de implementar.
4. Devolver ranking con: profit/hora recalculado, profit de referencia del Wiki, requisitos faltantes, timestamp de precios y advertencias de liquidez.

## 9. Ingesta del Wiki (v0: una sola vez)

- Fuente: dump del Wiki de OSRS. **[VERIFICAR]** la fuente oficial vigente de dumps; alternativa: API de MediaWiki paginando por categorías relevantes.
- Alcance del v0: páginas de quests, money making guides, guías de skills, ítems/NPCs/monstruos relevantes, mecánicas generales. Excluir redirects, namespaces de usuario y de discusión.
- Chunking **estructural**: por sección (encabezados), con solapamiento mínimo; tablas y plantillas convertidas a texto legible. Tamaño objetivo 300–600 tokens por chunk.
- Metadatos por chunk (`jsonb`): `page_title`, `section`, `page_type` (`quest`, `mmg`, `skill_guide`, `item`, `npc`, `other`), `url`, y cuando aplique `skills`/`quests` mencionados.
- Embeddings: locales, en batch; el texto de pasaje con prefijo `passage: `.
- **Licencia:** el contenido del Wiki es CC BY-NC-SA 3.0. Guardar `url` y atribución, mostrarlas en `sources`, y dejar un `NOTICE` en el repo.

### Esquema SQL (migración inicial)

```sql
create extension if not exists vector;

create table documents (
  id bigserial primary key,
  page_title text not null,
  section text,
  page_type text not null,
  url text not null,
  content text not null,
  metadata jsonb not null default '{}',
  embedding vector(384) not null,      -- ajustar a la dimensión del modelo elegido
  tsv tsvector generated always as (to_tsvector('english', content)) stored
);
create index on documents using hnsw (embedding vector_cosine_ops);
create index on documents using gin (tsv);

create table moneymaking_guides (
  id bigserial primary key,
  page_title text unique not null,
  data jsonb not null,                 -- campos de §8
  ingested_at timestamptz not null default now()
);

create table jev_judgments (           -- probabilidades crudas, para ajustar umbrales sin re-llamar a Jev
  id bigserial primary key,
  run_id text not null,
  kind text not null,                  -- route | relevance | sufficiency
  input_ref text,
  probs jsonb not null,
  created_at timestamptz not null default now()
);
```

La tabla `jev_judgments` se escribe solo en modo eval/debug, nunca con datos del jugador ni la key.

## 10. Plugin de RuneLite (Java)

- Basado en la plantilla oficial de plugins de RuneLite (Gradle). **[VERIFICAR]** la plantilla y la API de cliente vigentes antes de codear.
- UI: **un panel lateral** (`PluginPanel` + `NavigationButton`) con historial de chat, caja de texto y botón enviar. Texto simple/markdown básico. Sin overlays ni resaltado.
- Configuración (`Config`): URL del backend (default `http://localhost:8000`) y key de OpenRouter. **[VERIFICAR]** la forma recomendada de manejar un campo secreto en la config de RuneLite y los requisitos del Plugin Hub para plugins que envían datos a servidores externos.
- Lectura de estado: skills (`client.getRealSkillLevel`), quests (`Quest#getState(client)`), contenedores de inventario/banco (`ItemContainerChanged` / `client.getItemContainer`), tipo de cuenta por varbit. **Todo acceso a la API de cliente debe ocurrir en el client thread** (`clientThread.invokeLater`). Cachear el último snapshot; el banco se actualiza solo cuando se abre.
- HTTP: `OkHttp` (incluido en RuneLite) en un hilo aparte; nunca bloquear el client thread. Timeouts explícitos y mensajes de error legibles para `401/402/429/502`.
- `thread_id`: UUID generado al iniciar sesión del plugin; botón "Nueva conversación" que lo regenera.
- Sin automatización de acciones, sin interacción con el juego más allá de leer estado.

## 11. Configuración y seguridad

`.env.example`:

```
SUPABASE_DB_URL=postgresql://...
REASONING_MODEL=deepseek/...           # [VERIFICAR] slug vigente
JEV_MODEL=typesafe/jev-1.13
EMBEDDING_MODEL=intfloat/multilingual-e5-small
EMBEDDING_DIM=384
PRICES_USER_AGENT=osrs-agent-dev - contacto@...
GE_TAX_RATE=0.02
GE_TAX_CAP=
REL_THRESHOLD=0.70
SUFFICIENCY_THRESHOLD=0.50
CRAG_TOPK=20
MAX_CRAG_ITERATIONS=2
DEBUG_RESPONSES=false
```

Reglas:
- La key del usuario vive solo en memoria durante el request. Redactarla en logs, trazas y reportes de error.
- No usar una key propia del desarrollador en el backend salvo en scripts de ingesta o eval (variable de entorno local).
- Validar y limitar el tamaño de `player_state` y `message`.
- CORS cerrado; el backend local solo escucha en `127.0.0.1` durante el v0.

## 12. Evaluación

- `eval/questions.yaml`: 30–50 casos, cada uno con `question`, `intent_esperado`, `player_state` opcional y `expected_facts` (hechos que la respuesta debe contener o evitar). Incluir casos: quests con requisitos, moneymaking con distintos perfiles, flips, preguntas ambiguas, preguntas fuera de dominio, preguntas sin respuesta en el corpus, y consultas en español.
- `run_eval.py` reporta: precisión del router, precisión/recall del filtrado de pasajes, tasa de "no hay evidencia" correcta, latencia y costo (`usage.cost` de Jev y del LLM) por consulta.
- Con las probabilidades guardadas en `jev_judgments`, un script de ajuste recorre umbrales sin volver a llamar a Jev.

## 13. Orden de implementación sugerido

Cada hito tiene criterio de aceptación; no avances al siguiente sin cumplirlo.

1. **Esqueleto y clientes.** Repo, `config.py`, `prices.py` y `openrouter.py` con key por request, `jev.py` adaptando la Decisions API. *Aceptación:* script que clasifica 5 frases de ejemplo con Jev y obtiene precios actuales de 3 ítems; tests con mocks de HTTP.
2. **Tools deterministas.** Resolución de ítems, `calc_flip`, `rank_flips`, `get_price_history`. *Aceptación:* tests unitarios de la matemática (impuesto, márgenes, límites) con casos a mano.
3. **Ingesta.** Descarga/lectura del dump, parser, chunking, embeddings, carga a Postgres, y `moneymaking_guides`. *Aceptación:* conteo de documentos por `page_type`, y 10 guías de moneymaking inspeccionadas a mano contra el Wiki.
4. **Retrieval + grading.** Híbrido con RRF y `grade` con Jev en batch. *Aceptación:* con 10 preguntas del set de eval, los pasajes retenidos son relevantes (revisión manual) y el Noul de suficiencia separa bien los casos sin respuesta.
5. **Grafo completo.** Router, CRAG con corrección, workflows, memoria de hilo. *Aceptación:* conversación multi-turno por `curl` donde el segundo turno depende del primero ("¿y si no tengo ese ítem?").
6. **API HTTP.** `/v1/chat`, `/healthz`, errores tipados, redacción de la key. *Aceptación:* test que verifica que la key no aparece en ningún log.
7. **Plugin de RuneLite.** Panel, lectura de estado, cliente HTTP. *Aceptación:* en un cliente de desarrollo, enviar una pregunta que dependa del estado real del personaje y recibir una respuesta coherente.
8. **Evaluación y ajuste de umbrales.** `run_eval.py` completo y ajuste de `REL_THRESHOLD` / `SUFFICIENCY_THRESHOLD`. *Aceptación:* reporte con las métricas de §12.

## 14. Cosas a verificar antes de codear

- Request/response exacto de la Decisions API de Jev y cómo referenciar campos anidados en el `state`; contexto real (32K vs 64K).
- Slug vigente del modelo de razonamiento en OpenRouter y soporte de function calling del provider.
- Valores vigentes del impuesto del GE (tasa, tope, exenciones).
- Fuente y formato actual del dump del Wiki; plantillas reales de las money making guides.
- API vigente de RuneLite (nombres de enums/clases de contenedores, varbit de tipo de cuenta, detección de members) y requisitos del Plugin Hub sobre datos externos y secretos.
- API vigente de `langgraph-checkpoint-postgres`.
- Comportamiento de Jev con consultas en español (si degrada, traducir la query a inglés antes de clasificar).
- Reglas actuales de Jagex para plugins (solo lectura de estado, sin automatización).
