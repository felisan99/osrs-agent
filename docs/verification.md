# Verificación de los puntos **[VERIFICAR]** del SPEC

Fecha de la verificación: **2026-10-14**. Cada punto indica qué se comprobó, contra qué fuente y qué
cambió en el código. Lo que **no** se pudo verificar todavía está marcado como pendiente (con el hito
en el que se cierra), no como hecho.

## 1. Jev — Decisions API (OpenRouter)

| Tema | Resultado |
|---|---|
| Endpoint | `POST https://openrouter.ai/api/alpha/decisions`, `Authorization: Bearer <key del usuario>`. Existe además `POST https://openrouter.ai/api/v1/systemone` (forma nativa de TypeSafe, modelo `jev-1.13`), útil como plan B si el alpha cambia. |
| Modelo | `typesafe/jev-1.13` (fijo) o `~typesafe/jev-latest` (alias móvil). El campo `model` de la respuesta informa la versión exacta que respondió (p. ej. `typesafe/jev-1.13-20260917`). **Se fija la versión** porque los umbrales de §7 se ajustan contra sus probabilidades. |
| Request | `{model, state, questions: {<id>: {type, instructions, criteria}}}`. `state`: string, objeto o array. Los ids de pregunta son nuestros, no se envían al modelo y vuelven como claves de `answers`. |
| `noul` | `criteria` **debe** llevar `{"true": ..., "false": ...}` según la referencia de OpenRouter (los docs de TypeSafe dicen "opcional"). El adapter siempre envía ambas. Respuesta: `{type, noul: P(sí)}`, sin `confidence`. |
| `choice` | `criteria`: `{opción: descripción \| null}`, máx. 255. Respuesta: `{choice, probabilities, confidence}`. |
| `score` | `criteria`: lista de 2–10 niveles. Respuesta: `{score, legend, probabilities, confidence}`. |
| Respuesta | `{id, model, provider, usage{input_tokens, output_tokens, cost}, answers}`. |
| Errores | `{"error": {"code", "message", "metadata"}}` con 400/401/402/403/404/413/429/500/502/503/524/529. |
| Contexto | **32K tokens para `state` + la pregunta más larga; 64K para `state` + todas las preguntas** (docs de TypeSafe). OpenRouter anuncia 32K. Se presupuesta contra 32K (`JEV_MAX_STATE_TOKENS`, configurable). Como todas las preguntas ven el mismo `state`, el grading del CRAG cabe en **una** llamada: el `state` lleva los pasajes y cada Noul pregunta por uno. |
| Precio / límites | $0.042 por millón de tokens de entrada, salida gratis; 100K tokens/s y 80 req/s (aprox., ajustable por TypeSafe). |
| SDK | OpenRouter publica `openrouter` (Python, 1.3.50) pero exige `pydantic<2.13`, lo que choca con pydantic 2.14. TypeSafe publica `typesafe-sdk`. **Decisión:** adapter propio sobre `httpx` (`app/clients/jev.py`), aislando el formato alpha en un único módulo, como pide §3. |
| Referencia a campos | Los docs muestran referencias **de primer nivel** con backticks (`` `ticket` ``). Las rutas anidadas no están documentadas. **Resuelto en M4:** el `state` del grading usa campos planos (`passage_01`, `passage_02`, …) referenciados con backticks y funciona en vivo (27 casos, ver `docs/eval-m4.md`); no hizo falta probar rutas anidadas. |
| Español | TypeSafe: *"English is the primary training language… other languages are handled but not equally well"*. **Medido en vivo (2026-10-14, `scripts/m1_smoke.py`, 12 frases, un par EN/ES por intención): 12/12 correctas (6/6 EN, 6/6 ES), $0.000339 en total.** Pero hay señal de degradación: "¿Qué necesito para empezar Dragon Slayer II?" dio `quest` 0.63 (conf. 0.56) con `out_of_domain` en 0.35, una pregunta claramente en dominio. Con 12 casos no alcanza para estimar la tasa: se mide con el set de eval (M8). **Decisión de diseño:** rechazar como `out_of_domain` solo con probabilidad alta (`OUT_OF_DOMAIN_MIN_PROB`); un falso rechazo cuesta más que dejar pasar una pregunta ajena (el CRAG responde "sin evidencia"). La traducción previa a inglés queda como plan B si M8 muestra degradación. |
| Adapter | El adapter escrito a partir de los docs funcionó contra el API real en la primera llamada (request/response, key por request, `usage.cost`, `model` servido = `typesafe/jev-1.13-20260917`). |
| Riesgo conocido | Un paper de terceros (arXiv 2609.26758) reporta que el cabezal de decisión sigue el *nombre* de la opción más que su rúbrica. Mitigación: nombres de intención autoexplicativos (`app/graph/intents.py`) y descripciones con límites explícitos. |

## 2. Modelo de razonamiento (OpenRouter)

* `GET /api/v1/models` (462 modelos). Familia DeepSeek vigente: `deepseek/deepseek-v4-flash`, `-v4-pro`,
  `-v4.1-flash`, `-v3.2`… **todos** anuncian `tools` y `structured_outputs` en `supported_parameters`.
* **Default:** `deepseek/deepseek-v4-flash` (barato, contexto 1M, tools). Configurable (`REASONING_MODEL`).
  Los alias `~deepseek/…-latest` se evitan por la misma razón que `jev-latest`.
* Para garantizar function calling se envía `provider: {"require_parameters": true}` cuando hay `tools`.
* Errores de chat: forma `{"error": {code, message, metadata}}`; **OpenRouter puede devolver HTTP 200 con
  `error` en el cuerpo** o dentro de `choices[]` (se mapea igual); `Retry-After` en 429/503; un 402 sin
  `Retry-After` no se reintenta.
* **Medido en vivo:** `usage.cost` llega en chat. **El modelo "piensa" por defecto y factura esos tokens ocultos;
  con `max_tokens` bajo la respuesta puede quedar VACÍA** (visto: `finish_reason=length`, `content=""` con
  `max_tokens=200`; a 300 respondió). Mitigación: `reasoning: {"effort": "none"}` por defecto (`REASONING_EFFORT`,
  probado: respuestas directas, ~48 tokens vs ~109) y `max_tokens` por defecto de 1500 (`LLM_MAX_TOKENS`).
* **Function calling verificado** con los schemas reales de `app/tools/registry.py`: el modelo emitió dos
  `tool_calls` en paralelo (`get_item_price`, `calc_flip`) con argumentos válidos; `response_format=json_object` también funciona.
* El costo por llamada varía de ~$0.00001 a ~$0.00008 según el provider al que OpenRouter enrute.

## 3. Impuesto del Grand Exchange

Fuente: página *Grand Exchange* del Wiki (wikitext crudo) y `Module:Mmgtable.deduct_tax` (el propio Wiki).

* **2 %** desde el 29-may-2025 (antes 1 %), **redondeado hacia abajo por ítem**, **tope 5 000 000 por ítem**,
  cobrado **al vender**. Por debajo de 50 gp no se paga nada. Ejemplos oficiales: a 49 y a 50 el vendedor
  recibe 49; a 99 y a 100 recibe 98 (están en `tests/test_tax.py`).
* **Exentos:** bonos, Energy potion, flechas/dardos/runa de nivel bajo, comida de nivel bajo, ocho teletransportes,
  herramientas básicas. Las 45 entradas resuelven contra `/mapping` (test `live`). Ojo: en `/mapping` las
  tablillas se llaman `"<Ciudad> teleport (tablet)"` (salvo *Civitas illa fortis teleport*) y *Energy potion*
  es una familia de dosis (se expande a `(1)`–`(4)`).
* **Novedad (30-sep-2026, "Beyond Max Cash"):** el GE ya opera por encima de 2 147 483 647 con *platinum tokens*
  y los precios de guía se actualizan cada 3 h. En vivo ya existen precios > int32 (máx. visto: 9 364 999 989).
  → todo en enteros de Python; y el `gp` del jugador debe sumar `coins + 1000 × platinum tokens`
  (id 13204, valor 1000; **no** figura en `/mapping`, se agrega como ítem especial).

## 4. API de precios del Wiki

Verificada en vivo y con la página *Real-time Prices*:

* `/timeseries` **requiere `lookback`** (`6h|24h|7d|30d|6m|1y`); `timestep` devuelve 400. El spec era correcto.
* `/mapping`: 4 663 ítems; `limit` falta en ~11 % de ellos; **Coins y Platinum token no están**.
* `/latest`: 4 540 ítems, `Cache-Control: max-age=60`. `/1h`: ~3 100 ítems con promedios decimales.
* User-Agent bloqueados: `python-requests`, `Python-urllib`, `Apache-HttpClient`, `RestSharp`, `Java/x`, `curl/x`
  (+ `python-httpx` se rechaza por prudencia). Sin rate limit explícito; no iterar `/latest?id=`.
* La doc no define `high`/`low`; se usa la convención del spec (`high` = compra instantánea, `low` = venta instantánea).

## 5. Dump del Wiki y guías de moneymaking

* **No hay dump oficial publicado** (revisado: weirdgloop.org, docs del Wiki, búsqueda web). El Wiki indica
  `Special:Export`/API. **Decisión v0:** ingesta por la API de MediaWiki
  (`generator=categorymembers` + `prop=revisions`, 50 páginas por request, `User-Agent` descriptivo, secuencial).
* **Plantilla MMG:** de 649 páginas de `Category:Money making guides`, **641 usan `{{Mmgtable}}`** con
  parámetros consistentes (`Activity`, `Skill` con `{{SCP|Skill|nivel}}`, `Item`, `Quest`, `Members`, `Category`,
  `kph`, `isperkill`, `Input{N}`/`Input{N}num`/`Input{N}isph`/`Input{N}value`, `Output{N}`…). → **parser
  determinista con `mwparserfromhell`**; la pasada LLM de ingesta queda solo para los casos que el parser no cubra
  (≈ 12 páginas usan `{{#var:…}}`; 62 no tienen `kph`; 32 sin `Members`).
* Semántica (de `Module:Mmgtable`): solo las **salidas** descuentan impuesto (`deduct_tax`); con `isperkill` las
  cantidades son *por kill* y se multiplican por `kph` salvo `Input{N}isph`; `Output{N}value` / `Input{N}value`
  fijan el valor del ítem (nunca se grava); las guías `recurring` convierten con `Activity Time`
  (profit × 60 / minutos). Implementado y probado en M3 (`app/ingest/mmg.py`, `tests/test_mmg_parser.py`).
  `Exclude = yes` marca la guía como **obsoleta** (fuera de las listas del Wiki): el ranking la omite.
  Detalle y cobertura en [`docs/ingestion-m3.md`](ingestion-m3.md).
* El profit "histórico" no está en el wikitext (lo calcula Lua con precios vivos); `source_hourly_profit` se
  obtiene, como referencia, de la tabla renderizada de *Money making guide* y puede quedar `null`.
* **Licencia:** CC BY-NC-SA 3.0 → ver `NOTICE`. El carácter *NC* implica que el proyecto debe seguir siendo no comercial.

## 6. RuneLite

Verificado en `runelite/runelite` (master) y en `runelite/example-plugin`:

* `Client#getRealSkillLevel(Skill)`, `Quest#getState(Client)` → `QuestState{IN_PROGRESS, NOT_STARTED, FINISHED}`
  (coincide con §5), `Client#getItemContainer(int|InventoryID)`.
* `net.runelite.api.InventoryID` y `net.runelite.api.vars.AccountType` están **`@Deprecated`**. Usar
  `net.runelite.api.gameval.InventoryID` (`INV` = 93, `WORN` = 94, `BANK` = 95) y `VarbitID.IRONMAN` (1777).
* Tipos de cuenta de RuneLite: `NORMAL, IRONMAN, ULTIMATE_IRONMAN, HARDCORE_IRONMAN, GROUP_IRONMAN,
  **HARDCORE_GROUP_IRONMAN**` → el enum de §5 **no incluye** `hardcore_group_ironman`; se agrega. El mapeo
  valor-de-varbit → tipo se confirma en M7 contra el cliente de desarrollo.
* **Members:** no hay varbit/varp de membresía en `gameval`. `Client#getWorldType()` contiene `WorldType.MEMBERS`
  solo para el *mundo actual*. Regla: `true` si el jugador está en un mundo members (una cuenta free no puede
  entrar); si no, `null` y el agente pregunta cuando importe.
* `@ConfigItem(secret = true)` existe → campo de contraseña para la key. (El valor igualmente vive en el archivo
  de configuración local de RuneLite: documentarlo.)
* Plantilla: Gradle wrapper 8.10, `options.release = 11`, `runeLiteVersion = 'latest.release'`, `./gradlew run`
  abre el cliente de desarrollo. JDK local: 21 (JBR) → compatible.

## 7. Plugin Hub y reglas de Jagex

* El README del Plugin Hub **no** trae reglas explícitas sobre servidores externos ni secretos; solo "no
  malicioso, no romper reglas de Jagex, no ser una *Rejected/Rolled-back feature*", dependencias nuevas con
  verificación de hash y repositorio público. v0 **no** se publica (fuera de alcance), pero cualquier envío al Hub
  pasaría por revisión manual de la divulgación de datos.
* *Third-party client guidelines* de Jagex: solo clientes aprobados (RuneLite). Prohibido, entre otros:
  predicción del próximo ataque, indicadores de prayer, contadores de ataques, "indicar dónde pararse",
  información para *scoutear* objetivos PvP, nuevas entradas de menú que envíen acciones al servidor. Cláusula
  genérica: *"any features which act similarly… can also be considered unacceptable"*. Existe una excepción para
  "ayudantes externos accionados manualmente".
* **Implicación de diseño:** el agente es un chat accionado por el usuario, solo lectura, sin overlays ni menús;
  el prompt del sistema debe prohibir consejos tácticos de combate en tiempo real y cualquier automatización.

## 8. Pendientes de verificación

| Pendiente | Hito | Necesita |
|---|---|---|
| Plugin usado dentro del cliente de RuneLite **con una sesión iniciada** (criterio §13.7) | M7 | cuenta de Jagex: a mano, paso a paso en `docs/probar-el-plugin.md`. Lo que sí se verificó en un cliente real (RuneLite 1.13.1, perfil aislado, sin login): que el plugin carga y arranca (`OSRS Agent started`); eso encontró y corrigió un fallo de `./gradlew run` en macOS con Java 21 (`--add-exports java.desktop/com.apple.eawt`) |
| Usar el panel del plugin con el ratón y el teclado dentro de la ventana | M7 | una persona frente a la pantalla: no hay forma de manejar una ventana Swing desde este entorno |

Descartado por decisión del usuario (2026-10-14): medir la degradación del router en español con un set mayor;
la evaluación se concentra en inglés.

Cerrados desde la primera versión de este documento: referencias de campos en `state` de Jev (se usan campos
planos, `docs/eval-m4.md`); API de `AsyncPostgresSaver`, qué se persiste y que `Runtime[Context]` no persiste
nada (`docs/graph-m5.md`); tool calling con `effort: none` dentro del bucle completo (en vivo, flipping y
precios); `handleIteratedArgs` del módulo MMG (`docs/ingestion-m3.md`); valores de la varbit `IRONMAN`
(`Client#getAccountType()` ya devuelve el enum de RuneLite con los seis tipos, sin mapear varbits a mano).

## 9. Desvíos del SPEC introducidos después de los hitos

| SPEC | Qué se hizo | Por qué | Dónde |
|---|---|---|---|
| §2 Embeddings: modelo local multilingüe (default `multilingual-e5-small`) | default `Snowflake/snowflake-arctic-embed-s` (inglés, 384 dim, local) | la recuperación busca con la consulta **en inglés** que produce `condense`, así que no hace falta un modelo multilingüe; ganó en el benchmark de retrieval (desarrollo y reservado) y los modelos más grandes no ganaron | `docs/rag-quality.md` §5 |
| §9 Chunking: secciones primero, 300–600 tokens | tope de 480 tokens (el modelo acepta 512); un chunk nunca mezcla secciones de primer nivel; el encabezado es el camino que todo el chunk comparte | un chunk con el encabezado de otra sección mentía sobre su contenido | `docs/rag-quality.md` §2 |
| §9 Alcance de la ingesta | se agrega `Category:Strategies` (113 páginas) y una sección derivada "Drop sources" por ítem | las páginas de "cómo mato a X" no estaban, y la tabla de fuentes de los ítems la genera el Wiki al mostrar la página | `docs/rag-quality.md` §1–2 |
| §7 Memoria por hilo (checkpointer de Postgres) | los checkpoints se podan tras cada turno, hay un tope de 40 mensajes por hilo y los hilos inactivos se borran a los 30 días (`THREAD_TTL_DAYS`) | LangGraph no borra nunca un checkpoint | `docs/graph-m5.md` |
