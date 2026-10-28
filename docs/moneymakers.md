# `rank_moneymakers` (SPEC §8): reglas y validación

Fecha de las mediciones: **2026-10-14**. Código: `backend/app/tools/moneymakers.py`; tests con cifras
calculadas a mano: `backend/tests/test_moneymakers.py` (41); chequeo en vivo:
`uv run python -m scripts.check_moneymakers --crosscheck | --profile <perfil>`.

## Qué hace

1. **Elegibilidad** con el `player_state` (el modelo nunca lo pasa: lo inyecta el runtime y la tool no tiene
   argumento para él). Una guía se cumple si están todos sus requisitos duros: skills (nivel real), quests
   `FINISHED`, puntos de quest, combat level (calculado desde los skills con la fórmula del Wiki) y total level.
   Lo que el snapshot no informa es una **advertencia** (`caveats`), nunca un fallo: si el plugin no reportó
   una quest, no se la da por no hecha.
2. **"Casi elegibles"** (SPEC §8, "a pocos niveles"): a lo sumo 2 skills (o el combat level) faltándole hasta
   5 niveles cada uno y a lo sumo 1 quest sin terminar. Se listan aparte con lo que falta. Configurable
   (`MM_ALMOST_*`).
3. **Valuación con precios vivos**: `Σ(salidas/h × (precio − impuesto GE)) − Σ(entradas/h × precio)`.
   Impuesto solo sobre las ventas, por ítem, con tope y exentos (misma `TaxPolicy` que el flipping); los
   valores fijados por la guía (`value`) nunca pagan impuesto, igual que en `Module:Mmgtable`; las fórmulas de
   precio (`gep()`, `ha()`) y los `{{Cheap food}}` (el más barato hoy) se evalúan con los precios actuales.
4. **Liquidez**: una guía dice a qué ritmo se puede hacer, no si el mercado lo absorbe. Para los ítems que
   pesan ≥ 5 % del valor, el ritmo se limita a `MM_MARKET_SHARE` (25 %) del volumen horario de ese ítem y
   **toda la guía se escala** (la ganancia es lineal en el ritmo). La fila trae `pace` y una advertencia. Un
   ítem sin ningún trade en la última hora no se puede valuar y la guía se omite.
5. **Recurrentes aparte**: el Wiki da "profit por hora de tiempo activo" para actividades de pocos minutos
   cada tanto (granjas, compras diarias), lo que las pondría por encima de todo ("12 M/h" por comprar bastones
   cada 24 h). Se valúan **por vuelta** (`profit_per_run`, `active_minutes`, `repeats_every`) en `recurring`.
   La cifra del Wiki para ellas también es por vuelta (se comprobó: p. ej. granja de ranarr, 186,598 del Wiki vs.
   185,770 recalculado por vuelta).
6. **Ironmen**: no hay GP/h (decisión del usuario). Se devuelve `not_applicable_ironman` **sin ninguna cifra**
   y solo las guías escritas para ironmen ("(Ironman)") que cumplen; las demás son compras/ventas en el GE o
   valuaciones de botín, y solo se cuentan. `advice_pages` apunta a "Ironman money making guide" para que el
   flujo las recupere con CRAG.
7. **Capital**: la `{{Coins|N}}` de la línea *Item* de la guía ("10,000,000+ recommended") se guarda como
   `capital`; con `gp` informado, las guías que lo superan van a `needs_more_gp`.
8. **Exclusiones contadas** en `excluded`: obsoletas del Wiki (`Exclude=yes`), no calculables sin Lua
   (151 guías), solo members, requisitos sin cumplir, sin precio / sin volumen reciente.

## Base de precios (`MM_PRICE_BASIS`)

| Base | Salidas | Entradas | Mediana (recalculado / Wiki) |
|---|---|---|---|
| `mid` (default) | punto medio de `avgHigh` y `avgLow` | ídem | **1.000** (p10 0.985, p90 1.009) |
| `patient` | `avgHigh` (compra instantánea) | `avgLow` | 1.014 (p10 1.000, p90 1.053) |
| `instant` | `avgLow` | `avgHigh` | no medida; pesimista |

Se usa el promedio de `/1h` cuando ese lado operó en la última hora y, si no, el último trade de `/latest`
siempre que tenga menos de `MM_MAX_PRICE_AGE_S` (1 h). `mid` es el default porque **reproduce las cifras del
propio Wiki** (que usa su precio guía), y `patient` es optimista por el spread. El SPEC no fija la fórmula
más allá de "precio de venta/compra con `avgHighPrice`/`avgLowPrice`".

## Validación contra el Wiki (aceptación M3, punto 2)

`scripts.check_moneymakers --crosscheck` valúa las guías continuas para un jugador al máximo y compara con
`source_hourly_profit` (que el Wiki recalcula a diario con su módulo Lua), excluyendo las que el mercado
limita. Con `mid`: **102 guías comparables, mediana 1.000, 101 dentro de ±10 % y 102 dentro de ±25 %**; la
mayor diferencia (×0.83, "Smithing mithril dart tips") coincide con un precio que se movió desde el último
refresco del Wiki. Esto valida a la vez el parser (cantidades, `isperkill`, `isph`, fracciones, overrides) y
la valuación (impuesto, tope, exentos).

## Límites conocidos

* Las guías cuyo valor depende de módulos Lua del Wiki (drop tables, clues, `#switch`) no se pueden valuar con
  exactitud y quedan fuera (151 de 637; ver `docs/ingestion-m3.md`).
* El ritmo "realista" es un supuesto (25 % del volumen horario); las advertencias lo dicen y es configurable.
* Las quests de una guía son solo las que el Wiki enlaza como quests reales (los enlaces a páginas de
  estrategia se descartan en la ingesta); si el plugin nombra una quest distinto del Wiki, se informa como
  "no reportada", no como no hecha.
