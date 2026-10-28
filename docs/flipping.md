# Flipping: de "un ítem porque sí" a una recomendación justificada

Hasta ahora `rank_flips` ordenaba todo el mercado con **una foto** (últimos precios y volumen de la última hora)
por ganancia por ventana de 4 horas. Eso favorecía ítems de centavos con límite enorme (Ancient essence: 1 gp de
margen) y no podía responder "¿es un margen real o un pico?". El flujo de flipping ni siquiera tenía acceso al
historial. Esta versión (v0) le da al agente analítica sobre el historial y herramientas para justificar.

## Qué hace ahora

```
pregunta de flipping
  └─ flipping_workflow (modelo + herramientas, hasta 6 rondas)
       ├─ rank_flips(max_gp, strategy)   "¿qué flipeo?"
       ├─ analyze_flip(item, budget)     "¿vale la pena flipear X?"
       ├─ calc_flip(item, qty, precios)  la cuenta de UNA operación (impuesto, límite de compra)
       └─ get_price_history(item, lookback)
```

### `rank_flips`: dos etapas

1. **Foto del mercado** (`rank_candidates`): descarta lo que hoy no sirve (sin ganancia tras impuesto, sin liquidez,
   inasequible, cotización vieja, margen menor al mínimo de la estrategia) y deja hasta 40 candidatos.
2. **Historial** de cada candidato (7 días por hora; 30 días por 6 horas para los 12 mejores): `app/tools/flip_analysis.py`
   calcula, en código, los hechos que mira un flipper. Se piden unas 50 consultas a la API de precios del Wiki, con
   caché de 5 minutos y 10 en paralelo (+2 a 7 s por respuesta).

El resultado son pocas filas, cada una con **precios para ofertar**, unidades, ganancia por ventana, un **ejemplo
resuelto** que se puede verificar a mano, `evidence` (frases con los datos) y `flags` (riesgos).

### Qué se mide (por ítem)

| Dato | Para qué sirve |
|---|---|
| % de lecturas horarias con margen positivo tras impuesto | ¿el margen se repite o fue un instante? |
| Margen típico (mediana) y conservador (percentil 25) | cuánto esperar normalmente y en una mala hora normal |
| Volumen por hora del lado más lento (mediana y mala hora) | ¿se llena mi oferta?; no se confunde con un mercado fino |
| Ratio compra/venta | cerca de 1 = mercado equilibrado; muy lejos = un lado domina |
| Precio contra la mediana de 7 y 30 días, posición en el rango de 30 días, variación 24 h y 7 d | burbujas, hype y caídas |
| Horas estimadas para llenar | unidades / (volumen × 25 % que captura un flipper) |

Los precios que se recomiendan ofertar son los **habituales de la semana** (mediana), coherentes con una oferta
paciente; las últimas cotizaciones están en los extremos de la hora y exageran el spread. Un ítem con margen de 1 gp
no se recomienda: es un solo tick y cualquier movimiento lo borra (`min_margin` = 2 gp).

### Estrategias (`strategy`)

| Estrategia | Para quién | Umbrales |
|---|---|---|
| `volume` (por defecto) | ítems líquidos, muchas repeticiones, precio difícil de mover | ≥ 200 u/h, ROI ≥ 1 %, margen positivo ≥ 70 % de las horas; orden por ganancia × repetibilidad |
| `margin` | más margen por unidad, más lento, menos competencia | ≥ 20 u/h, ROI ≥ 3 %, ≥ 60 %; orden por ROI × repetibilidad |
| `overnight` | ofertas largas antes de desconectarse | ≥ 100 u/h, ventana de 8 h para llenar |
| `dip` | especulativo: líquido y por debajo de su precio habitual | compra al mínimo de hoy y vende a su precio habitual; ganancia conservadora = la mitad de la recuperación |

El agente usa `volume` salvo que le pidas otra cosa (decisión del usuario: preferir volumen a ítems poco
operados) y lo dice en la respuesta, con las otras opciones en una línea.

### Veredicto y confianza

`confidence_pct` multiplica: % de horas con margen positivo × equilibrio compra/venta × estabilidad (volatilidad) ×
distancia a la mediana × (hype cerca del máximo de 30 días). `verdict`: `solid` (≥ 75 %, sin avisos), `reasonable`
(≥ 50 %), `risky`, `avoid` (margen negativo, < 40 % de horas positivas o < 10 u/h).

## Referencia de la técnica

Los criterios salen de guías de flipping de terceros (comerciales, no oficiales): la
[guía de GE Tracker](https://www.ge-tracker.com/landing/osrs-flipping) (ítems de volumen y precio bajo, ratio
compra/venta cerca de 1, evitar mercados finos y burbujas en máximos, no invertir todo el banco) y
[OSRS Toolkit](https://osrstoolkit.com/osrs-merchanting-flipping-guide-2026/) (spread de ≥ 3 % antes de impuesto,
esperar tras lanzamientos, ofertas límite, dejar 20-30 % líquido). Están en el prompt como **principios para
explicar**, nunca como fuente de números: los números salen de las herramientas.

## Límites

* La historia no ve manipulación deliberada ni cambios de Jagex; la respuesta lo dice y no promete ganancias.
* La etapa 1 usa la foto actual: un ítem fiable que hoy no pasa el filtro de la foto no se analiza.
* Unidades por ventana = mínimo de presupuesto, límite de compra y el 25 % del volumen típico (supuesto, no medido).
* Pendiente (v1): un **backtest** (`eval/flip_backtest.py`) que mida si los ítems elegidos conservan su margen al
  día siguiente y permita calibrar umbrales y pesos con datos; y una guía de técnica indexada en el RAG para
  preguntas del tipo "¿cómo se flipea?".

Tests: `tests/test_flip_analysis.py` (historiales armados a mano) y `tests/test_tools_registry.py`; el eval tiene
`flip-justified-en` y `flip-dragon-bones-es`.
