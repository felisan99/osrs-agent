# OSRS Agent — plugin de RuneLite

Un panel lateral de chat que responde preguntas de Old School RuneScape (quests, formas de ganar GP, flipping,
precios, mecánicas) usando los datos de **tu** personaje. Es el cliente delgado del backend de este repo
(`../backend`); no hace ningún cálculo.

## Qué hace y qué no

* **Solo lee** el estado del juego, **solo cuando preguntás**, y nunca interactúa con el juego: sin overlays, sin
  resaltado, sin entradas de menú, sin automatización (SPEC §1, §10; pautas de clientes de terceros de Jagex,
  ver `../docs/verification.md` §7).
* Cada pregunta viaja con una foto compacta del personaje: tipo de cuenta, niveles reales, estado de las quests,
  puntos de quest, inventario, GP (monedas + fichas de platino), membresía si se puede saber, y el banco **solo
  si lo abriste** en esta sesión (con la fecha en que se capturó; se olvida al volver a la pantalla de login).
* La key de OpenRouter es un campo secreto de la configuración de RuneLite (`secret = true`); se envía al backend
  en el header `X-OpenRouter-Key` en cada pregunta y el backend la usa solo para esa pregunta, sin guardarla ni
  loguearla. RuneLite la guarda en tu perfil local (un archivo de propiedades en `~/.runelite`).
* **Legibilidad:** letra configurable (*Text size* 11-28, por defecto 15, y *Font*: Clean o la del juego; se
  aplica al instante) y las respuestas se dibujan por bloques (títulos, listas con sangría, citas y pie
  atenuados, atajos de preguntas en una conversación vacía) en vez de un bloque de texto. La respuesta nueva
  se muestra desde su primera línea. `AgentPanelTest` y `MarkdownLiteTest` fijan el comportamiento.
* Hilos: Swing en el EDT, todo acceso al cliente de RuneLite en el *client thread* (`ClientThread#invokeLater`) y
  HTTP con OkHttp en sus propios hilos, con timeouts explícitos y mensajes legibles para `401/402/429/502`.

## Datos que salen de tu máquina

Al backend configurado (por defecto `http://localhost:8000`, tu propia máquina): tu pregunta, el snapshot de
arriba y tu key de OpenRouter. El backend consulta a OpenRouter (con tu key) y al Wiki/API de precios; no
guarda tu snapshot (sí guarda el historial de la conversación por `thread_id`, que se regenera con "New
conversation").

## Compilar y probar

**Guía paso a paso, con qué escribir y qué debería pasar: [`../docs/probar-el-plugin.md`](../docs/probar-el-plugin.md).**
Desde la raíz del repo: `make backend` (terminal 1) y `make plugin` (terminal 2).

```bash
./gradlew test                    # 50 tests: lector del estado, cliente HTTP, markdown, panel, cableado de Guice
OSRS_AGENT_KEY=sk-or-... ./gradlew test --tests '*Live*'   # contra un backend real en marcha (gasta centavos)
./gradlew run                     # abre el cliente de RuneLite en modo desarrollador con el plugin
./gradlew panelPreview            # dibuja el panel a PNG en build/panel-preview, sin abrir RuneLite
```

`build.gradle` sigue la plantilla oficial (`runelite/example-plugin`, Gradle 8.10, Java 11, cliente
`latest.release`) con un agregado para macOS: la tarea `run` pasa
`--add-exports=java.desktop/com.apple.eawt=ALL-UNNAMED`. Sin eso, el cliente de RuneLite 1.13.1 no arranca con
Java 21 en un Mac (`IllegalAccessError ... com.apple.eawt.FullScreenAdapter` en `ClientUI.init`); el launcher
oficial pasa esa opción solo, Gradle no. Probado con RuneLite 1.13.1.

## Estado de la verificación

Verificado:

* compila contra la API vigente y pasan los tests (incluye el flujo completo del plugin con un servidor local y
  el cliente HTTP contra el backend real con una key real);
* **el plugin carga en un cliente real de RuneLite 1.13.1** (2026-10-14, con `./gradlew run`, sin iniciar sesión en el juego;
  esa corrida usó la carpeta `~/.runelite` habitual, no un perfil aparte: Java en macOS toma `user.home` del
  sistema y no de `$HOME`): el log del cliente muestra `Loaded plugin AgentPlugin`, `OSRS Agent started` y `Plugin
  AgentPlugin is now running`, y la configuración (`osrsagent.backendUrl`, `osrsagent.timeoutSeconds`) queda
  registrada. Esa prueba encontró el problema de macOS descripto arriba, ya corregido en `build.gradle`.

**No verificado en este entorno:** usar el panel con el ratón y el teclado dentro de esa ventana (no tengo forma de
manejar una ventana Swing), y el plugin con una sesión iniciada (necesita una cuenta de Jagex): el criterio de
aceptación del SPEC §13.7 (una pregunta que depende del personaje real, respondida con coherencia) queda para
correrlo a mano con la guía de arriba.
