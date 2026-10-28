# Calidad del RAG: chunking, plantillas y embedder

Fechas: **2026-10-14 y 15**. Código: `backend/app/ingest/` (`render.py`, `chunk.py`, `drops.py`), `backend/app/rag/`
(`embed.py`, `titles.py`, `retrieve.py`) y el benchmark `backend/eval/retrieval_bench.py`.

El punto de partida fueron dos observaciones sobre la primera ingesta: que el Wiki solo tiene sentido si cada
trozo de texto viaja con su título de página y su subtítulo ("Moss giant" › "Drops"), porque un ítem en una tabla
de drops no es lo mismo que un ítem mencionado en otra parte de la página (por ejemplo, una debilidad), y que no
estaba claro qué embedder se usaba ni si hacía falta uno mejor. Mirar los chunks reales mostró que el problema era
más amplio que la frontera entre chunks: **el índice perdía contenido** y **los encabezados mentían**.

## 1. Qué se encontró

Se inspeccionaron chunks reales (`Moss giant`, `Vorkath`, páginas de clues, ítems) y se auditó cada plantilla del
corpus (2,122 plantillas distintas se tiraban o se adivinaban; 66 estaban en la lista de descartadas):

| Hallazgo | Efecto |
|---|---|
| El chunk "Moss giant › Drops › 100%" contenía **toda** la tabla de drops (cada sub-tabla con su etiqueta), pero el encabezado nombraba solo la primera sección | un chunk decía ser "100%" y traía además la línea del `Mossy key`; los chunks cortos de secciones distintas ("Locations" + "Drops") se mezclaban |
| Las tablas de hierbas, semillas y gemas (`HerbDropLines`, `UncommonSeedDropLines`, `GemDropTable`, `RareDropTable`), el drop de **clue scrolls** (`DropsLineClue`) y las **notas de las tablas** (`raritynotes`: "con tarea de Slayer sube a 1/75") no se renderizaban | "¿los moss giants dan hierbas?", "¿y el mossy key con tarea?" no tenían respuesta en el índice |
| `Clue info` (**761 páginas** de pasos de clue) se descartaba entera | la página de "AN EARL" se indexaba sin la solución ("Speak to Ranael") |
| `Switch infobox` / `Multi Infobox` (**≈ 600 páginas**) y `Tabber` (251 páginas de ítems) se descartaban enteras | sin estadísticas de jefes con varias formas (Zulrah, Barrows), sin las recetas de las runas o las balas de cañón |
| `Recommended equipment`, `Equipment` e `Inventory` (los pasos centrales de las páginas de estrategia) | sin "qué equipo llevo a Vorkath" |
| La categoría `Strategies` (**113 páginas** `Jefe/Estrategias`) no estaba en el alcance | las páginas de "cómo mato a X" no existían en el índice |
| La tabla "Drop sources" de las páginas de ítems la genera el Wiki al mostrar la página (consulta a una base): **no está en el wikitext** | "¿qué monstruos dropean la dragon pickaxe?" no tenía respuesta: el dato solo vive en una fila de la tabla de cada monstruo |
| Un mensaje de más de 512 tokens **tumbaba la recuperación**: el embedder cambiaba `tokenizer.model_max_length` para acallar una advertencia y eso apagaba el truncado en sentence-transformers 6.1 | un jugador que pegue un texto largo (el límite de la API son 4,000 caracteres ≈ 1,000 tokens) recibía un 500; ningún caso del set de evaluación era tan largo |

## 2. Qué se cambió

**Chunking** (`app/ingest/chunk.py`):

* Un chunk **nunca mezcla secciones de primer nivel**: "Moss giant › Drops" y "Moss giant › Locations" son chunks
  distintos aunque sean cortos.
* Dentro de una sección, las sub-secciones se agrupan hasta el presupuesto, y el encabezado es **el camino que el
  chunk entero comparte** (`Moss giant › Drops`); cada sub-sección trae su etiqueta relativa (`[Weapons and armour]`).
  Si un chunk cae entero dentro de una hoja, el encabezado nombra la hoja (`Moss giant › Drops › Tertiary`).
* Una lista o tabla que continúa en el chunk siguiente **repite su línea de título o su fila de columnas**
  ("Items required:", "Level | XP | Tomes"), sin pasarse del presupuesto de 512 tokens.

**Plantillas** (`app/ingest/render.py`): las notas explicativas (`{{efn}}`, `{{Refn}}`, `<ref group=...>`) quedan
como `(note: ...)` donde aplican y las citas se siguen descartando; los infoboxes se agrupan por versión y con
etiquetas legibles, con la debilidad elemental y las defensas en una línea; `Tabber`/`<tabber>`/`Switch infobox`
conservan cada pestaña; las tablas compartidas de drops, los clues, el equipo y los inventarios recomendados, los
requisitos de runas, las tiendas y los monstruos por zona tienen su renderizador; las celdas que son fórmulas sobre
precios del GE (ruido: el dump solo tiene la fórmula) se reemplazan por `(depends on GE prices)`.

**Alcance y datos derivados:** se agrega `Category:Strategies` (113 páginas) y cada página de ítem recibe una
sección **"Drop sources"** que invierte las tablas de drops de los monstruos (`app/ingest/drops.py`: 21,552 filas
de 2,874 ítems); las cifras se copian de las tablas del Wiki, el orden usa la probabilidad parseada. Un jefe
nombrado en la pregunta trae también su página `/Strategies` a la lista por título.

Ejemplo, el chunk de la pregunta original (`Moss giant`):

```
ANTES   Moss giant > Drops > 100%            (el encabezado dice "100%", el chunk trae toda la tabla)
        [Drops > Seeds]
        35/128
        35/128                                (los nombres y la tabla de semillas se perdieron)
        - Mossy key: 1, rarity 1/150          (sin la nota: con tarea de Slayer es 1/75)
        (sin clue scroll, sin tabla de hierbas, sin tabla de gemas)

AHORA   Moss giant > Drops
        [100%] - Big bones: 1, rarity Always
        [Weapons and armour] - Black sq shield: 1, rarity 5/128 ...
        [Herbs] - Herb drop table: rarity 5/128
        [Seeds] - Uncommon seed drop table: rarity 35/128
        [Gem drop table] - Gem drop table: rarity 4/128
        [Tertiary] - Clue scroll (beginner): rarity 1/45
                   - Mossy key: 1, rarity 1/150, alternative rarity 1/75 (note: While having a moss giant
                     Slayer task, the drop rate is increased to 1/75, ...)
```

El índice pasó de 31,374 chunks (17,456 páginas) a 50,555 (17,561 páginas): más chunks, más chicos y cada uno
con el encabezado correcto.

## 3. Cómo se mide: el benchmark de retrieval

`backend/eval/retrieval.yaml` (79 preguntas de jugador, en inglés) y `backend/eval/retrieval_heldout.yaml` (24
preguntas **reservadas**, escritas después, sobre otras páginas y hechos, y que no se usaron para tocar nada;
una sola etiqueta se corrigió tras la primera corrida porque partía de un dato falso, ver la nota del caso
`ho-source-trident-of-the-seas`). No
llaman a ningún LLM ni a Jev, no necesitan key y corren en segundos:

```bash
make retrieval            # = cd backend && uv run python -m eval.retrieval_bench
make retrieval-heldout
uv run python -m eval.retrieval_bench --check-gold      # etiquetas vs el wikitext crudo
uv run python -m eval.retrieval_bench --only drops-moss-giant-list --verbose
```

Cada caso dice qué chunk cuenta como acierto: la **página** correcta, la **sección** correcta (`^Drops`, `^$` =
la intro) y el **texto** que contiene la respuesta (`Mossy key[^\n]*1/150`). Se reportan tres niveles (el
estricto exige los tres; el de contenido ignora la sección; el de página solo mira la página), el acierto en el
puesto 1/3/5/10/20 y el MRR. Las etiquetas se validan contra el **wikitext crudo** (`source:`), no contra el texto
ya renderizado, para que un error del renderizador no pueda dar por buena una etiqueta incorrecta.

Dos advertencias de honestidad: el set de desarrollo guió el trabajo (los casos de clues, estrategias y drops
nacieron para probar justamente lo que se arregló, y se agregaron algunos mientras tanto), de modo que **sus
números son optimistas por construcción**; los del set reservado son los que sirven como estimación. Y 79 + 24
casos no alcanzan para distinguir diferencias de uno o dos puntos: un caso vale ≈ 1.3 puntos en el set de
desarrollo y ≈ 4 en el reservado.

## 4. Resultados (índice completo, 50,555 chunks)

| | Set | @1 | @3 | @5 | @10 | @20 | MRR |
|---|---|---|---|---|---|---|---|
| **Antes** (índice anterior, e5-small) | desarrollo (79) | 46 % | 59 % | **67 %** | 72 % | **72 %** | 0.548 |
| | reservado (24) | 29 % | 50 % | **58 %** | 67 % | **71 %** | 0.411 |
| Contenido y chunking nuevos, **mismo embedder** (e5-small) | desarrollo | 63 % | 82 % | 91 % | 95 % | 97 % | 0.748 |
| | reservado | 38 % | 75 % | 83 % | 100 % | 100 % | 0.576 |
| **Ahora** (arctic-embed-s) | desarrollo | 66 % | 89 % | **92 %** | 95 % | **97 %** | 0.777 |
| | reservado | 46 % | 71 % | **88 %** | 100 % | **100 %** | 0.625 |

Dos lecturas: (1) **casi toda la mejora viene del contenido y del chunking, no del embedder**: con el mismo
e5-small, en el set reservado el acierto en los 5 primeros pasa de 58 % a 83 % y en los 20 primeros de 71 % a
100 %; (2) el cambio de embedder suma menos y de forma pareja: +0.03 de MRR en desarrollo y +0.05 en el reservado (en el
puesto 5: 91 % → 92 % y 83 % → 88 %). Al grader de Jev le llegan los 20 primeros, así que lo que más importa es el último renglón de cada bloque.

Por tipo de pregunta (set de desarrollo, acierto estricto):

| tipo | n | antes @5 | antes @20 | ahora @5 | ahora @20 | MRR antes → ahora |
|---|---|---|---|---|---|---|
| drops (monstruo → ítems) | 13 | 62 % | 62 % | 100 % | 100 % | 0.42 → 0.72 |
| sources (ítem → monstruos) | 10 | 60 % | 70 % | 100 % | 100 % | 0.61 → 0.95 |
| stats (debilidades, niveles) | 10 | 80 % | 90 % | 100 % | 100 % | 0.72 → 1.00 |
| items | 12 | 92 % | 92 % | 100 % | 100 % | 0.92 → 1.00 |
| quests | 10 | 80 % | 90 % | 80 % | 90 % | 0.47 → 0.47 |
| clues | 4 | 0 % | 0 % | 75 % | 75 % | 0.00 → 0.75 |
| strategies | 6 | 0 % | 0 % | 83 % | 100 % | 0.00 → 0.52 |
| mecánicas, lugares, diarios, skills, mmg | 14 | 86 % | 93 % | 86 % | 100 % | — |

Lo que sigue sin resolverse: las **quests** (los chunks `Details` compiten con la caminata; mismo valor antes y
ahora), y la búsqueda de un clue por **coordenadas** (`00 degrees 35 minutes north, 35 degrees 50 minutes east`):
son números exactos entre cientos de líneas casi idénticas y ni los embeddings ni el texto completo las separan.
Esos casos quedan en el benchmark como límites conocidos.

## 5. Qué embedder se usa y por qué

**Hoy: `Snowflake/snowflake-arctic-embed-s`** (33 M de parámetros, 384 dimensiones, entrada de 512 tokens,
licencia Apache 2.0, ≈ 130 MB), corriendo **en local** (GPU de Apple, sin gastar créditos de OpenRouter). Antes:
`intfloat/multilingual-e5-small` (118 M parámetros, 384 dimensiones), elegido en el hito 1 porque el SPEC pedía
un modelo local, chico y multilingüe.

El multilingüe dejó de hacer falta: el paso `condense` del grafo traduce la pregunta a inglés y la recuperación
busca **con esa consulta en inglés** (`state["query"]`); el modelo solo tiene que ser bueno en inglés. Las preguntas
en español siguen funcionando (comprobado en vivo con una conversación en dos idiomas y con los 25 casos en español
de la evaluación), con la salvedad de que dependen de que la traducción del LLM sea buena.

Se compararon modelos sobre **el mismo subconjunto del corpus** (2,777 páginas / ≈ 12,000 chunks: las páginas que
las etiquetas señalan, las que el recuperador confunde con ellas y una muestra al azar; `eval/subset_titles.py`),
porque embeber el Wiki completo con un modelo grande lleva horas. Un índice nuevo por modelo, con su propio
tokenizador y sus prefijos de consulta (`app/rag/embed.py`):

| Modelo | Parámetros | Dim. | @1 | @3 | @5 | @20 | MRR | ingesta del subconjunto |
|---|---|---|---|---|---|---|---|---|
| multilingual-e5-small (el anterior) | 118 M | 384 | 61 % | 81 % | 91 % | 97 % | 0.731 | 3.3 min |
| multilingual-e5-base | 278 M | 768 | 63 % | 81 % | 91 % | 96 % | 0.740 | 9.7 min |
| bge-small-en-v1.5 | 33 M | 384 | 63 % | 91 % | 92 % | 97 % | 0.763 | 2.8 min |
| **snowflake-arctic-embed-s** | 33 M | 384 | **66 %** | 89 % | 92 % | 97 % | **0.775** | 3.0 min |
| bge-base-en-v1.5 | 110 M | 768 | 65 % | 90 % | 92 % | 97 % | 0.776 | 8.3 min |
| snowflake-arctic-embed-m-v1.5 | 110 M | 768 | 63 % | 87 % | 92 % | 97 % | 0.758 | 8.5 min |

* Los modelos **en inglés** superan al e5 (+0.03 a +0.045 de MRR; el acierto en el puesto 3 sube de 81 % a 87–91 %);
  entre los cuatro en inglés no hay diferencia distinguible con 79 casos.
* **Crecer no ayuda**: pasar de small a base no mejora nada (e5: 0.731 → 0.740; arctic: 0.775 → 0.758; bge: 0.763 →
  0.776). El benchmark se satura en el 97 % del puesto 20 por los mismos 2–3 casos difíciles, sea cual sea el
  modelo. Por eso se eligió uno chico: mismas 384 dimensiones (el esquema no cambia), cuatro veces menos
  parámetros que e5-small y una ingesta completa de ≈ 11 a 14 minutos.
* Sobre el **índice completo** se repitió con los tres finalistas (tabla del §4 y de abajo): arctic-s fue el mejor
  o empató en los dos sets (desarrollo 0.777 vs 0.766 de bge-small vs 0.748 de e5; reservado 0.625 vs 0.586 vs
  0.576), y los números del subconjunto coincidieron con los del completo (0.777 en ambos), o sea que el
  subconjunto es un buen sustituto para comparar modelos.

Otros experimentos, todos sobre el subconjunto con e5-small salvo que se diga lo contrario:

| Qué | Resultado |
|---|---|
| Tamaño máximo de chunk 300 tokens (mínimo 150) | peor: MRR 0.695 vs 0.731 (480/300) |
| Mínimo 150 con máximo 480 | e5-small 0.756 (+0.025), arctic-s 0.773 vs 0.777: sin ganancia consistente; se mantiene 300/480 (SPEC) |
| Pesos de la fusión RRF (vector, texto, título) y la constante k | plano: MRR entre 0.67 y 0.72 en toda la grilla; los pesos iguales (el valor actual) están en la meseta |
| Reordenador (cross-encoder) | no se probó: a Jev le llegan los 20 primeros y el acierto en el puesto 20 es 97–100 % |
| `Qwen3-Embedding-0.6B` (estado del arte pequeño, 1,024 dimensiones, 600 M de parámetros) | **no gana**: MRR 0.756 (@1 65 %, @3 84 %, @5 90 %, @20 96 %) contra 0.777 de arctic-s, y embeber el subconjunto tardó 42 minutos contra 3 (≈ 4 chunks/s: una ingesta completa serían unas 3 horas). Descartado |

**Cómo cambiar de modelo** (el índice se reconstruye entero; los vectores de dos modelos no se pueden mezclar):

```bash
# 1. EMBEDDING_MODEL y EMBEDDING_DIM en backend/.env (o el default en app/config.py)
cd backend && uv run python -m scripts.ingest load --force      # ≈ 11 a 14 min con un modelo chico
uv run python -m eval.retrieval_bench                           # comprobar contra el benchmark
```

Salvaguardas: `scripts.ingest load` se niega a mezclar vectores de modelos o dimensiones distintos sin `--force`
(que vacía `documents`, recrea la columna para la nueva dimensión y re-ingiere); la tabla `index_meta` recuerda con
qué modelo se construyó el índice y **la API no arranca** si `EMBEDDING_MODEL` no coincide; el prefijo de cada
familia de modelos (E5, BGE, Arctic, Nomic, Qwen) sale del nombre y se puede pisar con `EMBEDDING_QUERY_PREFIX` /
`EMBEDDING_PASSAGE_PREFIX`.

## 6. Evaluación de punta a punta con el índice nuevo

`eval.run_eval` (servicios reales: Jev, el modelo de razonamiento, Postgres, las tools):

* **Set reservado: 12/13**, igual que antes, en las dos corridas (la única falla es la etiqueta ambigua
  `ho-fairy-quest-es`, que no se tocó).
* **Set de desarrollo: 43/46 la primera vez y 45/46 en la corrida final** (antes 46/46); el router sigue en 46/46,
  la respuesta queda retenida en 21/21 y no hay falsos "sin evidencia". Las tres fallas de la primera corrida eran
  **artefactos de los chequeos, no de las respuestas** (una comilla tipográfica en `Legends’ Quest`; una mención
  histórica y correcta del impuesto de 1 %; una respuesta honesta que cita el pet real). Se hicieron los chequeos
  más precisos, con tests que prueban que siguen atrapando los errores reales. La falla de la corrida final es la
  del pet inexistente: la respuesta dice que "Blorb" no existe y nombra el pet real de Zulrah, y el juez de Jev no
  la cuenta como "sin evidencia" porque trae un dato concreto; no inventa nada. Detalle en `docs/eval-m8.md`.
* Una observación que ningún chequeo cubre: en la respuesta de la corrida sobre el impuesto del GE el modelo
  extendió de más ("el comprador paga el precio más el impuesto; con 50 monedas, el comprador paga 51"). El Wiki
  dice otra cosa más acotada: por el redondeo, a un precio múltiplo de 50 el comprador paga una moneda de más.
  Se muestreó la pregunta siete veces más (cuatro con el índice nuevo, tres con el anterior): las siete repiten
  lo que dice el Wiki. Es una variación del modelo de generación, no del recuperador, pero conviene saberla.

## 7. Límites

* Son 79 + 24 preguntas de un solo autor: sirven para ver regresiones y diferencias grandes, no para estimar tasas.
* Las páginas fuera del alcance (NPCs que no son monstruos, tiendas, la mayoría de las subpáginas salvo
  `/Strategies`) siguen cayendo a la búsqueda en vivo (`search_live`), que es más lenta.
* "Drop sources" cubre lo que dicen las tablas de drops de las páginas ingeridas (21,552 filas de 2,874 ítems);
  fuentes que no son drops (tiendas, minijuegos, recompensas) no están en esa sección.
* Un cambio de plantilla del Wiki puede dejar de renderizarse sin avisar: `scripts` no tiene todavía una alarma que
  cuente las plantillas desconocidas más frecuentes; la auditoría de este documento se hizo a mano.
