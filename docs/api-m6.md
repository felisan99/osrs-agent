# Hito 6 — API HTTP: contrato y criterio de aceptación

Fecha: **2026-10-14**. Criterio del SPEC §13.6: *"errores tipados verificados; test que verifica que la key no
aparece en ningún log"*. Código: `backend/app/main.py` (FastAPI), `backend/app/service.py` (el agente como un
objeto), `backend/app/api/schemas.py` (el contrato).

```bash
cd backend
uv run python -m app.main            # escucha en 127.0.0.1:8000 (HOST/PORT en la config)
curl -s localhost:8000/healthz       # {"status":"ok","documents":50555,"guides":637,"quests":220}
curl -s -X POST localhost:8000/v1/chat \
  -H 'Content-Type: application/json' -H "X-OpenRouter-Key: $TU_KEY" \
  -d '{"thread_id":"mi-conversacion-01","message":"What do I need to start Monkey Madness II?","player_state":{"account_type":"main","members":true,"skills":{"slayer":66}}}'
```

## Contrato

* `POST /v1/chat`: header `X-OpenRouter-Key` (obligatorio), cuerpo `{thread_id, message, player_state?}`,
  respuesta `{answer, intent, sources[{title,url}], debug?}`; `debug` (`route_probs`, `retrieval_iterations`,
  `tools_called`) solo con `DEBUG_RESPONSES=true`.
* `GET /healthz`: 200 con el conteo del corpus, o 503 `unavailable` si Postgres no responde.
* `player_state` sigue el SPEC §5 con un agregado: `quest_points` (opcional; algunas guías exigen un total de
  puntos que no se puede derivar de la lista de quests). Se ignoran los campos desconocidos, los skills que no
  existen se descartan, un tipo de cuenta nuevo degrada a `unknown`, y `hardcore_group_ironman` (que el SPEC no
  listaba) está soportado.
* Errores tipados `{"error", "message", "retry_after_s"?}`:

| HTTP | `error` | Cuándo |
|---|---|---|
| 401 | `invalid_key` | falta el header o OpenRouter rechaza la key (el mensaje de OpenRouter, sin la key) |
| 402 | `insufficient_credits` | la cuenta del usuario no tiene créditos |
| 429 | `rate_limited` | límite de OpenRouter (con `Retry-After` y `retry_after_s`) |
| 502 | `upstream_error` | cualquier otra falla de OpenRouter/Jev/precios/Wiki |
| 413 | `payload_too_large` | cuerpo mayor que `4 × MAX_MESSAGE_CHARS + MAX_PLAYER_STATE_BYTES + 8 KB` (por cabecera y mientras se lee) |
| 422 | `invalid_request` | JSON mal formado, campo inválido (se nombra el campo, nunca el valor) o `message` de más de `MAX_MESSAGE_CHARS` |
| 500 | `internal_error` | cualquier otra cosa: mensaje genérico, sin detalles |

## Seguridad de la key (SPEC §1.5, §11) y cómo se verificó

* **El servidor no tiene key propia**: la key de desarrollo (`OPEN_ROUTER_KEY`) es solo de los scripts y ni
  siquiera es un setting del servidor. Llega en el header de cada request, se registra en el redactor de logs
  mientras dura, se entrega a los dos clientes que la necesitan (Jev y el modelo) y no vuelve a guardarse.
* **Logs**: acceso propio (método, ruta, estado, tiempo; sin headers ni cuerpos), el redactor se instala sobre
  los handlers del root y los errores tipados se convierten *dentro* del alcance de la key.
* **Superficie**: CORS cerrado salvo `CORS_ALLOW_ORIGINS`, `/docs` y `/openapi.json` apagados, escucha solo en
  `127.0.0.1`.
* **Tests** (`tests/test_api.py`, 25): contrato, cada error tipado, límites, y que la key no aparece en ningún
  log aunque el servicio intente imprimirla (`Authorization: Bearer …`, mensajes de error con la key, el
  traceback de un error inesperado); `tests/test_api_e2e.py` (con el modelo de embeddings real, marcado `live`)
  levanta el servicio real con OpenRouter y Jev simulados en la red y prueba que la key va **solo** a esas dos
  llamadas (header `Authorization`) y no aparece en la respuesta, en los logs ni en las tablas de checkpoints.
* **En vivo por curl** con la key de desarrollo: conversación de dos turnos con memoria del hilo, key inválida →
  `401 invalid_key` ("User not found." de OpenRouter), sin header → `401`, y **0 apariciones de la key** en el log
  del servidor.
* **Repetido con el índice nuevo** (2026-10-15, API real en `127.0.0.1:8000`): `/healthz` →
  `{"status":"ok","documents":50555,"guides":637,"quests":220}`; dos turnos (inglés con el snapshot de un jugador y
  seguimiento en español en el mismo hilo) respondidos con memoria, 0 apariciones de la key en el log y **un solo
  checkpoint** guardado para ese hilo al terminar.
* **Memoria de conversaciones**: tras cada turno el hilo conserva solo su último checkpoint, guarda como máximo 40
  mensajes y los hilos inactivos se borran a los `THREAD_TTL_DAYS` (30; 0 = nunca) al arrancar y cada 6 horas. Al
  arrancar, la API se niega a servir si `EMBEDDING_MODEL` no es el que construyó el índice (`index_meta`). Detalle
  en `docs/graph-m5.md` y `docs/rag-quality.md`.
