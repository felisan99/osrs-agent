# Hito 8 — Evaluación y ajuste de umbrales

Fecha: **2026-10-14**. Criterio del SPEC §13.8: *"reporte con las métricas de §12"*. Código:
`backend/eval/run_eval.py`, casos en `backend/eval/questions.yaml` (46) y `backend/eval/heldout.yaml` (13).

```bash
cd backend
uv run python -m eval.run_eval                          # los 46 casos, servicios reales (≈ $0.03, ≈ 2 min)
uv run python -m eval.run_eval --cases eval/heldout.yaml
uv run python -m eval.run_eval --check-gold             # etiquetas vs. el corpus cargado
uv run python -m eval.run_eval --sweep data/eval/run-<fecha>.json   # re-ajusta umbrales sin llamar a nada
```

Cada caso corre por el grafo real (Jev, el modelo de razonamiento, Postgres, las tools) en su propio hilo; los
turnos previos de un seguimiento se escriben en la memoria del hilo sin pagarlos. Los comportamientos
cualitativos ("dice que no encontró nada", "pide precisión") los juzga Jev con una pregunta Noul
(≈ $0.00001); el texto fijo de "sin evidencia" cuenta como ambos.

## Resultado

| | Set de desarrollo (46) | Set reservado (13) |
|---|---|---|
| Pasan todos los chequeos | **46/46** (tres corridas seguidas con la configuración final, y una cuarta con el último umbral) | **12/13** (dos corridas) |
| Precisión del router | 46/46 (en 28/28, es 18/18) | 12/13 |
| Recall del filtrado de pasajes (la respuesta queda retenida) | 21/21 | 5/5 |
| Precisión del filtrado (cota inferior: pasajes retenidos que contienen el dato) | ≈ 70 % con 3.4 pasajes retenidos por pregunta | 75 % |
| "Sin evidencia" correcto (preguntas que el Wiki no responde) | 6/6 | 2/2 |
| Ambiguas que piden más datos | 3/3 | 1/1 |
| "Sin evidencia" falso (preguntas con respuesta) | 0/25 | 0/6 |
| Hechos esperados / prohibidos / tools llamadas | 25/25, 4/4, 10/10 | 8/8, –, 4/4 |
| Latencia | mediana 4.3 s, p95 ≈ 10 s (las sin respuesta tardan ≈ 10 s: tres búsquedas) | mediana 3.6 s, p95 7.4 s |
| Costo por consulta | Jev $0.0005 + modelo $0.0001 ≈ **$0.0006** (el set completo: $0.028) | $0.0006 |

**Cómo leer estos números.** El set de desarrollo se usó para ajustar el router, los prompts y los umbrales
(abajo), así que su 100 % no es una estimación limpia; el **set reservado** son 13 preguntas escritas *después* de
ese ajuste y nunca usadas para tocar nada: 12/13, con el único fallo en una etiqueta discutible ("¿Qué quest
necesito para usar los fairy rings?" se enrutó a `quest` con 0.88, y la respuesta fue correcta; la etiqueta decía
`wiki_general`, y no se cambió después de ver el resultado). Son pocos casos y Jev no es del todo determinista
(las probabilidades varían unas centésimas entre corridas), de modo que el orden de magnitud es lo que cuenta:
~95 % de las preguntas pasan todos los chequeos, ninguna respuesta se da por "sin evidencia" cuando la había y
ninguna pregunta sin respuesta recibe una respuesta inventada.

## Lo que la evaluación encontró (y se corrigió)

El camino de las corridas: 42/45 → 42/46 → 46/46. Revisar **a mano las respuestas** (no solo los chequeos
automáticos) encontró lo más importante:

1. **Bug del parser de guías** (el más serio): un requisito duro escrito junto a una recomendación con `<br/>` en
   la misma viñeta se descartaba con ella ("{{SCP|Magic|55}} for High Level Alchemy<br/>…{{SCP|Prayer|43}}
   recommended…" → sin requisito de Magic). Un jugador free-to-play con 35 de Magic recibió "alquimia con acero"
   como método elegible. Se separan los segmentos por `<br/>` y se juzga cada uno: **34 guías más** con sus
   requisitos duros. El cruce de ganancias contra el Wiki (M2/M3) no lo podía ver: solo valida precios.
2. **El resumen del jugador ocultaba los nombres de las quests**: "¿puedo usar una dragon scimitar?" respondía
   "no puedo confirmar si terminaste Monkey Madness I" teniendo ese dato. Ahora el resumen lista el estado de
   las quests *nombradas en la pregunta o en la evidencia* (no manda las ~200 con cada pregunta).
3. **Inestabilidad del router en casos de borde**: `confidence` oscilaba entre 0.53 y 0.57 alrededor del piso de
   0.55 y la misma pregunta iba a `quest` o a `wiki_general` según la corrida; los seguimientos ("¿y cuál es su
   límite de compra?") se partían entre `price_lookup` y `flipping`. Cambios: los seguimientos se condensan
   **antes** de enrutar (el modelo ya se llamaba) y Jev recibe también la pregunta independiente; el piso de
   confianza baja de 0.55 a 0.45; y los criterios separan "cómo funciona el impuesto del GE" (wiki) de "cuánto
   impuesto pago por *esta* venta a *este* precio" (tool; el modelo la calculaba de cabeza). Una primera
   redacción sobre-corrigió (mandó "¿cómo funciona el impuesto?" a `flipping`); lo detectó la siguiente corrida.
4. **Formato**: tablas markdown, encabezados, reglas `---` y citas por número de pasaje (`[3]`) pese al prompt;
   el panel de RuneLite mostraría los símbolos tal cual. `clean_answer` los convierte (tablas → viñetas, citas →
   nombre de página).
5. Las respuestas de reescritura abrían con una disculpa ("Tienes razón, disculpa…") por el pedido de corrección;
   ahora el pedido dice "responde solo con la respuesta corregida".
6. **Etiquetas del set**: "¿Qué hechizo de Necromancy es el mejor para jefes?" no es una pregunta sin respuesta
   (el Wiki cubre los *thralls* del Arceuus spellbook, y el sistema los encontró), y "¿qué va a traer la próxima
   actualización?" tampoco (hay páginas de contenido anunciado, que `search_live` trae). Se reemplazaron por
   preguntas sin respuesta posible (la barra de habilidades de Revolution de RS3; un jefe secreto que Jagex no
   anunció) y la primera pasó a ser un caso con respuesta (`wiki-thrall-best-en`).

## Ajuste de umbrales (barrido offline sobre las notas guardadas de la corrida final)

`REL_THRESHOLD` (21 preguntas con respuesta; pasajes retenidos por pregunta y parte que contiene el dato):

| umbral | 0.3 | 0.5 | 0.6 | **0.7** | 0.8 | 0.9 |
|---|---|---|---|---|---|---|
| respuesta retenida | 21/21 | 21/21 | 21/21 | **21/21** | 21/21 | 17/21 |
| pasajes por pregunta | 7.3 | 6.1 | 5.0 | **4.0** | 3.0 | 1.8 |
| contienen el dato | 48 % | 52 % | 54 % | **64 %** | 73 % | 79 % |

Se mantiene **0.70** (el del SPEC): la respuesta sobrevive en toda la meseta 0.3–0.8 y a 0.9 se pierden 4 de 21, así
que 0.8 ahorraría contexto pero sin margen.

`SUFFICIENCY_THRESHOLD` (¿el veredicto coincide con "los pasajes retenidos contienen la respuesta"?):

| umbral | 0.3 | 0.4 | 0.5 | **0.6** | 0.7 | 0.8 |
|---|---|---|---|---|---|---|
| con respuesta → suficiente | 21/21 | 21/21 | 21/21 | **21/21** | 20/21 | 14/21 |
| sin respuesta → insuficiente | 8/9 | 8/9 | 9/9 | **9/9** | 9/9 | 9/9 |

Pasa de **0.50 a 0.60**: la meseta perfecta es 0.5–0.6, a 0.4 una pregunta sin respuesta pasa por suficiente y a
0.7 se pierde una con respuesta; 0.50 era el borde de la meseta y el error caro es el falso "suficiente" (deja
pasar una pregunta sin respuesta). Con el hito 4 (otro set, 27 casos) la separación era 0.70 vs 0.28.
`ROUTE_MIN_CONFIDENCE` pasa de 0.55 a **0.45** por la inestabilidad descrita arriba. Quedaron como parámetros
(`REL_THRESHOLD`, `SUFFICIENCY_THRESHOLD`, `ROUTE_MIN_CONFIDENCE`); la corrida guarda los valores usados.

## Límites de esta evaluación

* 46 + 13 casos: sirve para detectar regresiones y errores sistemáticos, no para estimar tasas con precisión.
* El juez de comportamientos es Jev (el mismo modelo que decide el resto); para "sin evidencia" casi todos los
  casos se resuelven por el texto fijo, sin juicio.
* El español se midió con 18 + 7 preguntas: sin degradación visible del router (100 % y 6/7, el fallo es la
  etiqueta ambigua). El SPEC pedía medirlo con un set mayor; **se decidió no ampliarlo** (2026-10-14): la
  evaluación se concentra en inglés, que es el idioma del Wiki y el de la mayoría de las preguntas. Lo que se
  amplía es el set en inglés (retrieval: `docs/rag-quality.md`).
* No hay evaluación del plugin dentro del juego (ver `plugin/README.md`).

## Re-medición con el índice nuevo (2026-10-15)

Después del trabajo de calidad del RAG (`docs/rag-quality.md`: contenido, chunking y embedder nuevos) se repitió la
evaluación completa, sin cambiar umbrales. Dos corridas del set de desarrollo: la primera con el índice recién
construido y la final con todo el código terminado (incluida la regla de las preguntas sin sujeto, ver
`docs/graph-m5.md`).

| | Antes | Primera corrida | **Corrida final** |
|---|---|---|---|
| Desarrollo (46): todos los chequeos | 46/46 | 43/46 (3 artefactos de los chequeos, abajo) | **45/46** |
| Router | 46/46 | 46/46 | 46/46 (inglés 28/28, español 18/18) |
| Respuesta retenida | 21/21 | 21/21 | 21/21 |
| "Sin evidencia" correcto / falsos | 6/6 / 0 | 6/6 / 0 | 5/6 / 0 (el caso del pet inexistente, abajo) |
| Preguntas ambiguas que piden detalle | 3/3 | 3/3 | 3/3 |
| Reservado (13): todos los chequeos | 12/13 | 12/13 | **12/13** (la misma etiqueta ambigua, `ho-fairy-quest-es`) |
| Latencia (desarrollo) | mediana 4.3 s, p95 ≈ 10 s | 6.4 s / 16.5 s (con una ingesta corriendo en la misma máquina) | mediana 5.4 s, p95 12.6 s |
| Costo por consulta | ≈ $0.0006 | ≈ $0.0007 | ≈ $0.0006 (Jev $0.00040 + modelo $0.00019) |

**Las tres fallas de la primera corrida** (`data/eval/run-rag-arctic-dev.json`) eran artefactos de los chequeos:

1. `quest-ds2-reqs-en`: la respuesta nombraba **Legends’ Quest** con apóstrofe tipográfico y el chequeo buscaba
   `Legends' Quest`. Ahora el comparador de hechos normaliza comillas (con un test).
2. `wiki-ge-tax-en`: prohibía la cadena "1%", pero el índice nuevo trae la historia del impuesto y la respuesta dice,
   correctamente, que *"era 1 % hasta el 29 de mayo de 2025 y subió a 2 %"*. Ahora prohíbe presentar el 1 % como
   tasa vigente ("es / actualmente / ahora … 1 %") y un 5 % suelto.
3. `none-fake-pet-en`: prohibía cualquier fracción `1/N`; la respuesta dijo que no existe el pet "Blorb" y citó el
   pet real de Zulrah con su tasa real (1/4,000). Ahora prohíbe una fracción en la misma frase que el nombre
   inventado.

Esos cambios se hicieron después de ver el resultado, y por eso se dejan escritos acá; cada uno tiene un test que
comprueba que el chequeo sigue atrapando el error real y deja pasar la respuesta honesta.

**La falla de la corrida final** es el mismo caso `none-fake-pet-en`, ahora por el chequeo de *comportamiento*:
Jev (el juez) no cuenta como "sin evidencia" una respuesta que dice *"no existe ningún pet Blorb en el Wiki; el pet de
Zulrah es el Pet snakeling, 1/4,000"* porque además da un dato concreto (de otro pet). No inventa nada, que es lo
que ese caso protege, y es más útil que el texto fijo de antes; el caso pasa o falla según cómo salga redactada la
respuesta (pasó en la primera corrida). No se tocó el criterio del juez para no ajustar el examen a la respuesta.
