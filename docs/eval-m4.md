# Hito 4 — Retrieval híbrido + grading con Jev: criterio de aceptación

Fecha: **2026-10-14**. Criterio del SPEC §13.4: *"con 10 preguntas del set de eval, los pasajes retenidos
son relevantes (revisión manual) y el Noul de suficiencia separa bien los casos sin respuesta"*.

Se corrió con **27 casos** de `backend/eval/questions.yaml` (los que se responden con pasajes del Wiki y
tienen respuesta conocida: 18 con respuesta en el corpus, 9 sin respuesta o ambiguos; los 2 seguimientos
multi-turno se saltan porque dependen del paso que condensa la pregunta, M5). Corpus: **17,456 páginas /
31,374 chunks** cargados en Postgres. Modelo de decisión: `typesafe/jev-1.13-20260917`.

```bash
cd backend
uv run python -m scripts.m4_check                         # recuperación + grading con Jev (usa OPEN_ROUTER_KEY)
uv run python -m scripts.m4_check --retrieval-only        # solo recuperación: sin Jev, sin key
uv run python -m scripts.m4_check --oracle-rewrite        # busca con la reescritura a inglés escrita a mano
```

Las probabilidades crudas quedan en `backend/data/eval/m4-<fecha>.json` para re-ajustar umbrales sin volver a
llamar a Jev.

## Resultado

| Métrica | Resultado |
|---|---|
| Jev separa los casos sin respuesta | **Sí, con hueco limpio**: P(suficiente) de los casos respondibles ≥ **0.70** (media 0.87, n=16) frente a ≤ **0.28** de los demás (media 0.13, n=11). `SUFFICIENCY_THRESHOLD=0.50` queda en el medio. Acierta 27/27. |
| Pasajes retenidos relevantes (revisión manual) | Los 27 casos revisados a mano (ver abajo): los retenidos son del tema de la pregunta; promedio **2.9** pasajes retenidos de 20. |
| El pasaje con la respuesta llega a los candidatos | **16/18** con la pregunta tal cual; **18/18** con una reescritura perfecta a inglés (`--oracle-rewrite`). |
| …y sobrevive al filtro de relevancia | 16/16 de los recuperados. |
| Costo de Jev | **$0.00038 por pregunta** ($0.0103 las 27), una sola llamada por pregunta. |
| Latencia | recuperación ≈ 0.3 s (mediana), grading ≈ 2.3 s (mediana; máx. 3.3 s) con 20 candidatos de ≈ 500 tokens. |

Los dos casos que no se recuperan son preguntas **en español sin nombres del juego en inglés** ("hueso de
dragón", "gárgolas"): el vector multilingüe no alcanza solo y la rama de texto completo es inglesa. Jev los
juzgó insuficientes (0.16), que es lo correcto con lo recuperado: es la señal que dispara el bucle CRAG
(`rewrite` a inglés → `retrieve` otra vez, **M5**). Con la reescritura perfecta ambos se recuperan.

### Recuperación: antes y después de los cambios del hito

Rango del primer pasaje que contiene la respuesta, sobre las 18 preguntas respondibles, preguntas originales:

| Variante | En top-20 | Top-10 | Top-5 | MRR |
|---|---|---|---|---|
| Base (`ts_rank_cd`, palabras ≥ 3 letras, sin pesos) | 15/18 | 11 | 9 | 0.44 |
| **Final** (cabecera ponderada A, `ts_rank`, palabras de 2 letras, lista por título) | **16/18** | **15** | **15** | **0.69** |
| Final + reescritura perfecta a inglés | 18/18 | 17 | 16 | 0.74 |

Qué se cambió y por qué (cada cambio se midió sobre el mismo set, offline y sin costo):

1. **Cabecera ponderada en el índice de texto** (`supabase/migrations/20261014020000_weighted_tsv.sql`). Cada
   chunk empieza con `Página > Sección`; esas palabras pesan A y el cuerpo D. Sin esto, "requisitos de Monkey
   Madness II" no encontraba el chunk que *es* la lista de requisitos (es una lista de nombres de skills y
   quests que no se parece a nada). Se usa `ts_rank` (mejor MRR que `ts_rank_cd` en este corpus).
2. **Lista por título** (`app/rag/titles.py`): si la pregunta nombra una página ("Monkey Madness II",
   "Cook's Assistant", "gargoyles"→"Gargoyle"), los chunks de esa página forman una tercera lista de RRF,
   ordenados por cercanía vectorial. Coincidencia exacta normalizada (sin acentos ni apóstrofes, el nombre más
   largo gana, plural→singular). Las palabras sueltas genéricas ("Combat", "Items", …) no cuentan.
3. **Palabras de 2 letras** en la consulta de texto ("II" distingue Monkey Madness II de I; "XP", "GP"); las
   palabras vacías de 2 letras se filtran en Python para que "or" no se lea como operador.

### Hallazgo de implementación: segfault con embeddings concurrentes

Con dos preguntas evaluadas a la vez, el proceso moría con **SIGSEGV** (exit 139): `SentenceTransformer.encode`
no tolera llamadas simultáneas desde varios hilos en el backend MPS de Apple, y el API embebe la consulta en
hilos (`asyncio.to_thread`). Habría tumbado el servidor con dos requests concurrentes. Arreglo: un candado
alrededor de cada `encode` (`app/rag/embed.py`), con test de regresión (`tests/test_embed_concurrency.py`,
verificado que falla sin el candado). Una consulta tarda milisegundos, así que serializar no pesa.

## Umbrales (barrido offline sobre las probabilidades guardadas)

`REL_THRESHOLD` (pasajes con el dato / retenidos por pregunta; el dato es una cota inferior de la precisión
porque otros pasajes pueden ser legítimamente relevantes):

| umbral | 0.3 | 0.5 | 0.6 | **0.7** | 0.8 | 0.9 |
|---|---|---|---|---|---|---|
| preguntas con el dato retenido | 16/18 | 16/18 | 16/18 | **16/18** | 16/18 | 16/18 |
| pasajes retenidos por pregunta | 5.7 | 4.4 | 3.7 | **2.9** | 2.5 | 2.1 |
| parte que contiene el dato | 48 % | 55 % | 59 % | **66 %** | 67 % | 73 % |

`SUFFICIENCY_THRESHOLD`: cualquier valor entre 0.3 y 0.7 clasifica bien los 27 casos (a 0.8 se pierden 3
respondibles). **Se mantienen los valores iniciales del SPEC (0.70 / 0.50)**: están dentro de una meseta
amplia, así que con 27 casos no hay evidencia para moverlos. Se vuelven a ajustar en M8 con el set completo
y el flujo extremo a extremo.

## Revisión manual de los pasajes retenidos

Leídos los 27 casos (informe completo: salida de `m4_check`). Resumen:

* **Respondibles (18):** el pasaje que contiene el dato queda retenido en 16/16 de los recuperados y es
  siempre el primero o está entre los primeros por relevancia (0.91–0.99 el mejor de cada pregunta). Los demás retenidos son del mismo tema (otras filas de la tabla
  de armas, el mismo dato en otra página: `Ancient Magicks`/`Spellbook`/`Magic`, o la guía de dinero de
  Barrows para "¿dónde queda Barrows?").
* **Falsos positivos de relevancia visibles:** "¿Cómo lo consigo?" retuvo una sección de `Haunted Mine`
  (0.71, apenas sobre el umbral) y "¿Cuál es la mejor forma de entrenar?" retuvo 7 pasajes de entrenamiento
  free-to-play. En ambos la **suficiencia** quedó baja (0.12 y 0.28), que es la que decide: ahí el agente debe
  pedir precisión, no responder.
* **Sin respuesta (nombre inventado, dato fuera del Wiki, juego equivocado; los 6 casos `none-*`):** 0
  retenidos y suficiencia ≤ 0.17. **Ambiguos (3):** suficiencia 0.12–0.28.

## Notas de diseño que se confirmaron

* La suficiencia se pregunta en la **misma** llamada que la relevancia (SPEC §7: una sola llamada batch), o
  sea sobre el conjunto de candidatos y no sobre el retenido; para que no haya contradicción `Grader` exige
  además que haya al menos un pasaje retenido. En los 27 casos nunca discreparon.
* Referencias a campos de `state` en las preguntas de Jev: se usan **campos planos** (`passage_01`, …) con
  backticks; funcionó en vivo, no hizo falta probar rutas anidadas.
* Los resultados dependen de que el set de eval tenga etiquetas correctas: `eval/cases.py::check_gold` valida
  las páginas y los fragmentos de evidencia contra el corpus cargado (y los tests validan el formato del YAML).

## Pendiente (hitos siguientes)

* **M5:** paso que condensa/traduce la pregunta a inglés (seguimientos y español) y bucle de corrección
  (`rewrite`, `search_live`). Después de eso se vuelve a medir la recuperación con la reescritura real del LLM
  (hoy se mide el techo con `--oracle-rewrite`).
* **M8:** set completo extremo a extremo, ajuste final de umbrales y tasa real de degradación en español.
