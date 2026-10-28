# Hito 5 — El grafo (LangGraph): diseño y criterio de aceptación

Fecha: **2026-10-14**. Criterio del SPEC §13.5: *"conversación multi-turno por curl donde el turno 2 depende
del 1 (p. ej. «¿y si no tengo ese ítem?»)"*. Se cumplió en vivo, contra Jev y el modelo reales, y se repitió por
HTTP en el hito 6 (`docs/api-m6.md`).

## Forma del grafo

```
start_turn → classify_intent ─┬→ refuse (out_of_domain) ───────────────────────────────┐
                              └→ condense ─┬→ quest_prepare ──────┐                     │
                                           ├→ moneymaking_prepare ┼→ retrieve → grade ─┐│
                                           ├→ (wiki_general) ─────┘        ↑           ││
                                           ├→ flipping ──────────────────────┼───────┐ ││
                                           └→ price_lookup ──────────────────┘       │ ││
              grade ─┬→ generate ───────────────────────────────────────────────────→ finalize → END
                     ├→ no_evidence ────────────────────────────────────────────────→ ┘
                     └→ correct (≤ MAX_CRAG_ITERATIONS) → grade
```

* **`classify_intent`** (Jev Choice): el estado lleva el mensaje, un resumen corto de los turnos anteriores y una
  línea del jugador. Confianza baja → `wiki_general`; `out_of_domain` solo con probabilidad ≥ 0.80 (en vivo, una
  pregunta legítima de Dragon Slayer II en español llegó a 0.35 de `out_of_domain`). El rechazo es un texto fijo,
  sin llamar al modelo.
* **`condense`**: el mensaje como consulta independiente en inglés más el idioma de la respuesta. Resuelve los
  seguimientos ("¿y si no tengo ese ítem?") con el historial y traduce el español a los términos del juego. Solo
  llama al modelo cuando hace falta: un primer mensaje claramente inglés es su propia consulta.
* **CRAG** (`retrieve → grade → [correct → grade]* → generate | no_evidence`): una llamada de Jev por ronda
  (relevancia por pasaje + suficiencia). `correct` pide al modelo otra consulta en inglés (sin repetir las
  fallidas), busca de nuevo sin restricción de tipo, agrega `search_live` (API de MediaWiki → chunks → ranking por
  embedding) y re-evalúa conservando lo que ya era relevante. Tras `MAX_CRAG_ITERATIONS` correcciones sin
  evidencia suficiente: *"no encontré información confiable"*, sin inventar.
* **`quest_workflow`**: identifica la quest nombrada (numerales y apóstrofes normalizados: "Monkey Madness 2"),
  compara **de forma determinista** sus requisitos con el snapshot (skills con niveles que faltan y si son
  *boostables*, quests en el orden sugerido, puntos de quest, combat level, ítems en inventario/banco con el
  banco "no capturado" como incertidumbre) y fuerza las páginas de la quest en la recuperación.
* **`moneymaking_workflow`**: `rank_moneymakers` (`docs/moneymakers.md`) + CRAG sobre las guías recomendadas. Sin
  snapshot (jugador sin sesión iniciada) igual se rankea: todas las guías por GP/h con precios del momento, cada
  una con sus `requisitos`, y se avisa que el ranking no está personalizado; lo que el jugador escribió acota:
  una categoría ("solo combate") y un nivel de combate ("lvl 80 combat"), que filtra por combate.
  Cada fila lleva `you_meet` (requisitos ya verificados contra el personaje, p. ej. "Perilous Moons (quest
  finished)"), `wiki_requirements` (el texto de la guía, que mezcla requisitos con niveles *recomendados* y equipo) y
  la lista `close_to`: una frase por guía "casi elegible" con lo que falta, redactada por el código. El modelo
  arma la respuesta con esos datos (qué podés hacer ya / qué te falta para lo siguiente), sin plantilla fija, y
  no puede dar por faltante una quest o nivel que no esté en `missing`. Una quest que la guía pide solo
  "Started" cuenta si está en curso. Ironmen: sin cifras de GP/h, ni siquiera las que cite una
  página del Wiki (`profit_rate_mentions` lo verifica y pide reescribir o descarta la frase).
* **`flipping` / `price_lookup`**: el modelo elige tool y argumentos y las tools hacen todo el cálculo. El flipping
  tiene `rank_flips` (dos etapas, con historial y estrategias), `analyze_flip`, `calc_flip` y `get_price_history`, y
  debe justificar con los datos de las filas (`docs/flipping.md`). La primera
  ronda tiene que llamar a una tool; si la API de precios cae, un texto fijo reemplaza la respuesta; ironmen:
  "el flipping no aplica" sin llamar al modelo.
* **Si hay un resultado determinista** (ranking o chequeo de quest) y las páginas no alcanzan, igual se responde
  con él (un ranking de moneymaking ni gasta correcciones): solo "sin evidencia ni resultado" termina en
  `no_evidence`.

## Principios hechos cumplir por código, no solo pedidos al modelo

| Principio (SPEC §1/§7) | Cómo se garantiza |
|---|---|
| Los números los calcula el código | Cada número ≥ 1,000 de una respuesta con precios/ganancias debe aparecer en las tools, en lo que escribió el jugador o en los pasajes (`grounding.py`); si no, una reescritura y, si persiste, se descarta la frase. Medido sobre formatos reales (`1,500,000`, `1.5M`, `2.3 millones`). |
| Indicar el timestamp de los precios | El pie `Precios … obtenidos <fecha> UTC` lo agrega el código (`footer.py`), no el modelo. |
| Sin evidencia no se inventa | Texto fijo tras agotar las correcciones; el prompt prohíbe completar con memoria. En vivo: "¿Cuál es la comida favorita de Zulrah?" → 3 búsquedas → "no encontré información confiable". |
| La tasa del impuesto no se cita de memoria | `calc_flip`, `rank_flips` y `rank_moneymakers` devuelven `tax_rate_pct`; un test en vivo detectó al modelo diciendo "1 %" y se agregó. |
| La key es un secreto | Ver abajo. |

## Qué se persiste y qué no

El estado del grafo (lo que LangGraph guarda en Postgres tras cada paso) lleva **la conversación** y los datos de
trabajo del turno (consulta, pasajes del Wiki, resultados de las tools). La **key del usuario y el snapshot del
jugador viajan en el contexto de la corrida** (`RunContext`, `context_schema` de LangGraph): no son parte del
estado ni de la config, así que nunca llegan a un checkpoint. `tests/test_graph.py` lo verifica escaneando todas
las tablas de LangGraph tras dos turnos, con una key y un snapshot con marcas únicas, y comprobando que el escáner
sí ve lo que se guarda (la conversación). Lo que sí queda en los checkpoints son resultados deterministas del
turno calculados *a partir* del snapshot (p. ej. el chequeo de requisitos de la quest consultada), no el snapshot.

Verificado contra LangGraph 1.2.15 / `langgraph-checkpoint-postgres` 3.1.3 (2026-10-14):
`AsyncPostgresSaver(pool).setup()` crea sus tablas; el pool debe abrirse con `autocommit=True`,
`prepare_threshold=0` y `row_factory=dict_row`; de la config solo `thread_id` y claves de ids se copian a la
metadata del checkpoint; `Runtime[Context]` entrega el contexto a los nodos sin persistirlo.

## Preguntas sin sujeto: se pregunta, no se busca

Un primer mensaje que es solo una pregunta con pronombre ("¿y dónde se consigue?", "where do I get it?": 6
palabras o menos, un `it/that/this/its` y ninguna página nombrada) va de `condense` directo a `no_evidence`, que
pide más detalle, sin buscar. Hasta entonces dependía de que Jev juzgara insuficientes los pasajes que devuelve una
búsqueda de "cómo conseguir algo" sin ítem; en una de cinco corridas los juzgó suficientes y el agente contestó
sobre un ítem cualquiera (un antipoison). Un seguimiento nunca cae en esta regla: la conversación dice qué es "it".
`tests/test_graph_units.py` fija los casos que sí y los que no (el borde: "tell me about that weapon from abyssal
demons" busca normalmente).

## Saludos y agradecimientos: respuesta fija, sin costo

Un mensaje compuesto **solo** por saludo, gracias o despedida ("Hello!", "hola", "gracias!", "chau"; 5 palabras o
menos, todas del vocabulario de `app/graph/smalltalk.py`) se contesta con un mensaje fijo en el idioma usado
(qué puede hacer el asistente), sin búsqueda ni llamadas a Jev o al modelo. Antes caía en `out_of_domain` y recibía
el rechazo pensado para pedidos de bots. "Hi, what is a whip?" tiene más palabras y sigue el camino normal. La
intención devuelta es `smalltalk`.

## Cuánto crece la memoria: límites (`app/graph/retention.py`)

LangGraph escribe un checkpoint por cada paso de cada turno y nunca borra uno (`AsyncPostgresSaver` implementa
`delete_thread` pero no `prune`). Nada de este agente lee historial, un turno solo necesita el **último** estado
del hilo, así que hay tres límites:

| Límite | Dónde | Efecto |
|---|---|---|
| Un checkpoint por hilo | `prune_thread`, al terminar cada turno (`AgentService.chat`) | se borran los checkpoints, escrituras pendientes y blobs de versiones reemplazadas; solo queda lo que usa el último. Un fallo al podar se loguea y **nunca** rompe la respuesta |
| 40 mensajes por hilo | `MAX_STORED_MESSAGES` (`graph/state.py`) | el reductor de `messages` conserva los 40 más recientes (20 intercambios); los prompts usan solo los últimos 12 (`history_max_messages` en `config.py`) |
| Hilos inactivos | `delete_stale_threads`, al arrancar la API y cada 6 horas (`THREAD_TTL_DAYS`, 30 por defecto; 0 = nunca) | se borra todo hilo cuyo último checkpoint tenga más de N días |

Con eso el tamaño de la memoria queda acotado por (hilos activos × un checkpoint de ≤ 40 mensajes). A mano:

```bash
cd backend
uv run python -m scripts.maintenance stats                       # hilos, checkpoints, tamaño
uv run python -m scripts.maintenance prune --older-than-days 7   # limpia una base que creció antes
```

Medido sobre la base de desarrollo (2026-10-14): 311 hilos con 3,006 checkpoints, 6,926 blobs y 14,211 escrituras
→ 311 checkpoints, 3,110 blobs y 2 escrituras. `tests/test_retention.py` comprueba sobre Postgres real que la
poda deja exactamente un checkpoint, que el hilo sigue recordando (el turno siguiente ve los anteriores), que otro
hilo no se toca y que un hilo viejo se borra y uno reciente no.

## Aceptación en vivo (dev key, ≈ $0.001 por turno)

* **Turno 1**, perfil `main_midgame`: *"What do I need to start Monkey Madness II?"* → intención `quest`, chequeo
  determinista (Crafting 62/70, The Eyes of Glouphrie sin empezar, Recipe for Disaster "verificar en el juego",
  ítems con el banco sin capturar), $0.0005.
* **Turno 2**, mismo hilo: *"¿y si no tengo el m'speak amulet?"* → el paso `condense` lo resuelve con el turno 1,
  recupera las páginas del ítem y responde en español cómo conseguirlo de nuevo, $0.0007.
* Otros caminos: moneymaking con ranking vivo y pie con timestamp, flipping ("2 % de impuesto, −76 gp por
  unidad"), precio de la whip, rechazo de un pedido de bot (sin llamar al modelo), pregunta sin respuesta, ironman
  sin GP/h, pregunta en español sobre "hueso de dragón" (que en M4 no se recuperaba) respondida con las 72 XP.

## Límites conocidos

* Una conversación no puede "pedir" que se conserve su historia más allá de los 40 mensajes: lo anterior se olvida
  (por diseño; ver la tabla de límites).
* El idioma "other" cae al español por defecto; el set de eval mide la degradación real en M8.
* `search_live` funciona contra la API real (≈ 1.5 s por consulta; además de las páginas ingeridas trae páginas
  `Update:` de cambios recientes, p. ej. el cambio del impuesto del GE, que no están en el corpus). Su efecto
  sobre la calidad de las respuestas se mide en M8.
