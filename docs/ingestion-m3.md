# Hito 3 — Ingesta del Wiki: criterio de aceptación

> **Actualizado el mismo día (calidad del RAG):** la renderización de plantillas, el chunking (encabezados que
> reflejan la sección real del chunk), el alcance (se suma la categoría `Strategies`), las secciones derivadas
> "Drop sources" y el modelo de embeddings cambiaron después de este hito; `docs/rag-quality.md` explica qué y por
> qué, con números. Los conteos de abajo son los de la **primera** ingesta (31,374 chunks).

Fecha: **2026-10-14**. Criterio del SPEC §13.3: *"conteo de documentos por `page_type`; 10 guías de
moneymaking inspeccionadas a mano contra la página del Wiki"*.

## Cómo se ingiere

No existe un dump oficial del Wiki de OSRS (`docs/verification.md` §3), así que la ingesta pagina la API de
MediaWiki por categorías (`generator=categorymembers`, 50 páginas por request, `User-Agent` descriptivo,
`maxlag`, pausa entre requests). Todo es **local y reanudable**: `fetch` baja el wikitext a
`backend/data/wiki/*.jsonl` (una categoría terminada se salta) y `load` parsea, parte en chunks, embebe con
el modelo de embeddings en local (entonces `multilingual-e5-small`; hoy `snowflake-arctic-embed-s`, ver
`rag-quality.md`) (**sin gastar créditos de OpenRouter**) y escribe Postgres; una página con la
misma revisión no se vuelve a procesar.

```bash
cd backend
uv run python -m scripts.ingest fetch --scope v0         # ~2 min para las 18.7k páginas
uv run python -m scripts.ingest load  --scope v0         # ~8 min en un MacBook (MPS), ~60 chunks/s
uv run python -m scripts.ingest load  --scope v0 --page-type mmg --force   # re-parsear solo las guías
uv run python -m scripts.ingest stats
```

Alcance `v0` (`app/ingest/scope.py`; la primera categoría que lista una página decide su `page_type`; entre
paréntesis, páginas descargadas sin redirects): Quests (228), Miniquests (35), Quick guides (207), Money making
guides (649), Training guides (92), Guides (89), Skills (35), Bosses (173), Slayer monsters (687), Mechanics
(179), Combat (43), Ironman Mode (57), Prayers (62), Transportation (45), Slayer (181), Minigames (53),
Achievement diaries (95), Slayer Masters (25), Raids (5), Grand Exchange (6), Player-owned house (45),
Treasure Trails (185), y a granel Monsters (1,636), Spells (253), Locations (1,181) e Items (12,495). Una página
en varias categorías se guarda una sola vez, por eso las páginas cargadas son menos que la suma.

## 1. Conteo por `page_type`

Carga completa, **0 errores, 0 chunks por encima de las 512 tokens** del embedder:

| `page_type` | chunks | páginas |
|---|---:|---:|
| `item` | 15,010 | 12,372 |
| `other` | 5,991 | 2,144 |
| `npc` | 3,610 | 1,640 |
| `quest` | 2,559 | 464 |
| `skill_guide` | 2,945 | 187 |
| `mmg` | 1,259 | 649 |
| **Total** | **31,374** | **17,456** |

Tamaño de los chunks (caracteres; ≈ 3.6 por token): mediana 890–1,260 según el tipo, p95 ≤ 1,650, máximo
2,183. Cada chunk empieza con `Página > Sección` (título y encabezado entran al embedding y al índice de
texto). El tope es 480 tokens y no los 600 del SPEC porque el embedder trunca en 512 (desviación documentada en
`Settings.chunk_target_max_tokens`).

Además de los chunks, la ingesta extrae datos **estructurados** que usan las tools deterministas:
**637 guías de moneymaking** (tabla `moneymaking_guides`) y **220 quests** con requisitos de skills, quests,
ítems y recomendados (tabla `quests`).

## 2. Diez guías de moneymaking contra el Wiki

`uv run python -m scripts.inspect_guides --n 10 --seed 7 --stratify` imprime cada guía parseada junto al
wikitext del que salió (mezcla recurrentes, por-kill, por-hora y no soportadas). Comparadas a mano:

| Guía | Tipo | Qué se comprobó | Resultado |
|---|---|---|---|
| Farming torstol | recurrente | Actividad 6 min cada 90, farming 85 y magic 61, quests (Ghosts Ahoy, My Arm's Big Adventure, Fairytale I), ítems de entrada y salida | ✔ Los opcionales ("Making Friends with My Arm", "Dream Mentor") y el Agility *recomendado* no se cuentan como requisitos |
| Farming huasca | recurrente | ídem con farming 65 | ✔ |
| Buying battlestaves from Zaff | recurrente | 840,000 coins → 120 battlestaves cada 1,440 min; el valor `Input1value = 1` fija las monedas a 1 gp | ✔ |
| Collecting jangerberries | por kill | 20 inventarios/h × 28 bayas; `Input2num = 1/8` → 0.125 anillos | ✔ fracciones y `isperkill` |
| Tanning blue dragonhide | por kill | 3,500 pieles/h; entradas sin cantidad = 1; `1/26` energy potions = 0.0385; tarifa en coins | ✔ |
| Cutting maple logs | por hora | sin `isperkill`: 360 logs **por hora**; requisito Woodcutting 45 (el "60+ recomendado" no cuenta) | ✔ |
| Making guam potions | por kill | 4,000/h, Herblore 3, Druidic Ritual | ✔ |
| Looting in Revenant Caves | por hora | entradas y `Output1 = Coins 600000` con valor fijo; requisitos solo "recomendados" → ninguno | ✔ pero ver hallazgo 1 |
| Crafting blood runes at Arceuus (minimal requirements) | por kill | cantidades con `{{#switch:{{min|…}}}}` y variables | ✔ marcada **no soportada** con la razón (no se adivina) |
| Creating forester's rations | por kill | `{{#invoke:RationProfitCalculator|…}}` | ✔ marcada **no soportada** (ítems y cantidad calculados por Lua) |

Hallazgos de la inspección:

1. **`Exclude = yes` significa "obsoleta"** (`Template:Mmgtable/doc` y `Module:Mmgtable`: no se escribe en
   Bucket, queda fuera de las listas del Wiki y se categoriza como *Obsolete money making guides*). "Looting in
   Revenant Caves" la tiene, y su salida es una estimación fija de 600,000 coins para un botín que el propio
   texto dice que va "de cientos de coins a millones según suerte". Se agregó `MoneyMakingGuide.obsolete`
   (**23 guías**, 18 de ellas valorables) y el ranking las omite y cuenta la omisión.
2. Las guías **sin profit de referencia** (100 de las valorables) no son un fallo del parser: el Wiki solo
   lista las "worthwhile" (profit por encima de ~15,000–20,000 gp/h; `Module:Mmgtable`). Con precios vivos se
   valoran igual, pero el número de referencia no existe.
3. Las cantidades de las guías **no soportadas** se muestran como `x1` por defecto: están marcadas y nunca
   entran en un ranking (`MoneyMakingGuide.rankable`).

## Cobertura del parser de guías

* 649 páginas → **637 parseadas** (las 12 restantes no tienen tabla de guía), **0 caídas**.
* **486 totalmente soportadas**, **468 tras descartar las obsoletas** (73 % de las guías).
* **151 no soportadas**, siempre con la razón guardada. Causas (una guía puede tener varias): ítems sin precio en
  el GE 191 (casi todos conceptuales: "Rare drop table" 34, "Gem drop table" 34, "Spirit seed" 16, claves y
  clues), valores calculados por módulos Lua del Wiki (`Average drop value` 57, `RDTValue` 39, `GDTValue` 34,
  `SeedPackValue` 16, valores de clues 13), `#switch`/variables que dependen de Lua 44, `#invoke` 9. Solo 10
  guías quedan fuera únicamente por precios faltantes.
* Decisión de diseño: **no se reproduce la lógica Lua** del Wiki. Una guía cuyo valor no se puede calcular de
  forma exacta se marca no soportada y no se rankea; es preferible a mostrar un número inventado (principio 1
  del SPEC).

## Verificación automática

`tests/` cubre el renderizador de wikitext, el parser de guías (39 tests, con fixtures reales del Wiki), el de
quests, el chunker (invariante de 512 tokens), el cliente MediaWiki, el cargador a Postgres y la CLI. La
validación contra el corpus real es esta inspección más `ingest stats`.
