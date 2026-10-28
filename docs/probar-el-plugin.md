# Cómo probar el plugin, paso a paso

El plugin es un cliente delgado: un panel de chat dentro de RuneLite que le manda tu pregunta (y una foto
compacta de tu personaje) al backend de este repo. Se prueba en tres niveles; cada uno cubre algo que el
anterior no puede.

| Nivel | Qué cubre | Qué necesita | Tiempo |
|---|---|---|---|
| **1. Automático** | lector del estado, cliente HTTP, panel y cableado de Guice, con un servidor local simulado | nada más | 1 min |
| **2. En RuneLite, sin iniciar sesión** | que el plugin carga en el cliente real, el panel, la configuración y la comunicación con el backend | tu key de OpenRouter | 10 min |
| **3. En RuneLite, con tu personaje** | que la respuesta usa tus niveles, quests y GP reales (criterio de aceptación del SPEC §13.7) | una cuenta y el paso de credenciales de la §4 | 15 min |

Todo se lanza desde la raíz del repo con `make` (`make help` lista los comandos). Para el día a día hay un
lanzador de un clic que arranca y apaga todo: [`docs/lanzador.md`](lanzador.md).

## 1. Automático (sin pantalla)

```bash
make test-plugin        # = cd plugin && ./gradlew test
```

Son 39 tests de JUnit; 2 de ellos (`LiveBackendTest`) solo corren con `OSRS_AGENT_KEY=sk-or-... ` y contra un
backend vivo. Esto **no** abre RuneLite: no prueba que el plugin cargue en el cliente real.

## 2. En RuneLite, sin iniciar sesión

### 2.1 Levantar el backend (terminal 1)

```bash
make backend            # arranca Postgres si hace falta y la API en http://127.0.0.1:8000
```

Dejá esa terminal abierta. En otra, comprobá que responde:

```bash
curl -s localhost:8000/healthz     # {"status":"ok","documents":...,"guides":...,"quests":...}
```

Si falla: la API no está corriendo, o Postgres no arrancó (`make db`).

### 2.2 Abrir el cliente con el plugin (terminal 2)

```bash
make plugin             # = cd plugin && ./gradlew run
```

La primera vez descarga y compila (1–3 minutos); después se abre una ventana de RuneLite con el plugin cargado
desde el código fuente, en modo desarrollador y con logs de depuración en esa terminal. Es el cliente normal:
llega hasta la pantalla de login de Old School. Señal de que cargó bien, en esa terminal:

```
... PluginManager - Loaded plugin AgentPlugin
... AgentPlugin - OSRS Agent started
... PluginManager - Plugin AgentPlugin is now running
```

(Esos tres renglones los verifiqué en un cliente real. Si el cliente muere con `IllegalAccessError ...
com.apple.eawt`, falta `--add-exports` en la tarea `run` de `plugin/build.gradle`: ya está puesto.)

**Qué es normal en esa terminal** (lo vas a ver y no es un fallo):

* La barra de Gradle se queda en `<===========--> 85% EXECUTING ... > :run` todo el tiempo. Es el cliente
  corriendo: la tarea termina recién cuando lo cerrás.
* Cientos de líneas `DEBUG ... EventBus - Registering class ...`, y las de tus otros plugins (por ejemplo
  los del Plugin Hub). Es el modo `--debug`. Para ver solo lo nuestro: `make plugin 2>&1 | grep -i osrsagent`.
* Un `DEBUG ... ReflectUtil - null` con un stack trace (`InaccessibleObjectException` o
  `NoSuchFieldException: root`): lo emite RuneLite al arrancar con Java 21; no afecta al plugin.
* `Ping - sequence number mismatch` / `recvfrom() error ... errno 35`: el plugin World Hopper midiendo la
  latencia de los mundos.
* `[LWJGL] ... Unsupported JNI version detected`: aviso del plugin GPU en macOS.

Un fallo real sería una línea `ERROR` o `WARN` que nombre `com.osrsagent`, o que falte alguno de los tres
renglones de arriba.

* Usa tu perfil de RuneLite de siempre (`~/.runelite`). Si preferís no mezclar la configuración, creá antes un
  perfil nuevo desde RuneLite (Configuration → Profiles).
* Para cerrarlo, cerrá la ventana o `Ctrl+C` en esa terminal.

### 2.3 Configurar el plugin

1. En la barra lateral derecha, la llave inglesa (**Configuration**).
2. Buscá **OSRS Agent** y abrí su engranaje.
3. Completá:
   * **OpenRouter API key**: tu key (`sk-or-...`). Es un campo secreto; RuneLite la guarda en tu perfil local
     (por eso conviene una key con tope de gasto y vencimiento).
   * **Backend URL**: `http://localhost:8000` (el valor por defecto).
   * **Answer timeout**: 120 s alcanza (las respuestas de flipping tardan de 14 a 17 s).
   * **Text size** (11-28, por defecto 15) y **Font** (*Clean*, una sans-serif legible, o *RuneScape*, la del
     juego): se aplican al instante, sin reiniciar, y la conversación se vuelve a dibujar.

Para ver cómo queda el panel **sin abrir RuneLite**: `cd plugin && ./gradlew panelPreview` genera PNG en
`plugin/build/panel-preview/` (conversación completa con 15 y 18 puntos, con la fuente del juego y el estado vacío).

### 2.4 Hacer las preguntas

En la barra lateral derecha aparece un ícono naranja con forma de globo de chat (tooltip **OSRS Agent**).
Un clic abre el panel. Probá, en orden, y compará con lo que debería pasar:

| # | Escribí | Debería pasar |
|---|---|---|
| 1 | `What is the drop rate of the mossy key from moss giants?` | Aviso gris *"Not logged in ... answering without your character data"*. La respuesta dice 1/150 y que sube a 1/75 con tarea de Slayer de moss giants, y cita la página **Moss giant**. |
| 2 | `¿Qué necesito para empezar Monkey Madness II?` | Responde **en español**, con los requisitos de la quest y sus fuentes. |
| 3 | `¿Cuánto cuesta una abyssal whip hoy?` | Un precio con la **hora** del dato al pie (los precios vienen del API del Wiki, no del modelo). |
| 4 | `¿Y dónde se consigue?` (justo después de la 3) | Entiende que habla de la abyssal whip: memoria del hilo. |
| 5 | **New conversation**, y luego `¿Y dónde se consigue?` | Ya no sabe de qué hablás y te lo pide ("No encontré información confiable sobre eso... ¿Podés darme más detalle?"): el hilo nuevo no recuerda, y una pregunta sin sujeto no se busca. |
| 6 | `Dame una receta de lasaña` | Rechazo breve (lo decide Jev, sin llamar al modelo de razonamiento): solo responde de Old School RuneScape. |
| 7 | `Hello!` / `Hola` | Saludo y qué puede hacer el asistente (no un rechazo). Sin costo. |
| 8 | `¿Cuál es la mejor forma de ganar dinero con un personaje de lvl 80 de combate, solo con combate?` | Sin sesión: avisa que no conoce tu personaje, lista guías de combate con GP/h del momento **y sus requisitos**, y el precio con hora al pie. Con sesión iniciada filtra por tus stats reales. |

Si una respuesta tarda, la barra de abajo dice **Thinking...**; lo normal son 4–10 segundos.

### 2.5 Qué dice el plugin cuando algo no anda

| Mensaje en el panel | Qué pasó | Qué hacer |
|---|---|---|
| *Set your OpenRouter API key in the plugin settings...* | no hay key | cargala en la configuración |
| *Could not reach the backend at ...* | la API no está corriendo o la URL está mal | `make backend`; revisá **Backend URL** |
| *OpenRouter did not accept your API key* | key inválida o borrada | revisá la key en openrouter.ai |
| *Your OpenRouter account has no credits left...* | sin saldo (HTTP 402) | cargá crédito |
| *OpenRouter is rate limiting you...* | límite de velocidad (HTTP 429) | esperá lo que indica |
| *The backend did not answer within N seconds* | demora o proveedor caído | subí **Answer timeout** o reintentá |

Dónde mirar: la terminal del backend (una línea por pedido: método, ruta, estado y milisegundos; **nunca** la
key), la terminal de `make plugin`, y `~/.runelite/logs/client.log`.

## 3. Sin RuneLite: el backend solo

Para separar un problema del plugin de uno del backend:

```bash
cd backend
uv run python -m scripts.chat --profile main_midgame \
    --say "How do I start Monkey Madness II?" --say "¿y si no tengo el m'speak amulet?"
```

Usa la key de desarrollo del `.env` (no la del plugin), imprime qué decidió el grafo (intención, búsquedas,
herramientas, costo) y arma el personaje con un perfil de prueba. Es lo mismo que recibe la API del plugin.

## 4. En RuneLite, con tu personaje

Necesita una sesión iniciada. El cliente de desarrollo no usa el Jagex Launcher por sí solo; el procedimiento
oficial de RuneLite ([Using Jagex Accounts](https://github.com/runelite/runelite/wiki/Using-Jagex-Accounts)) es:

1. Tené el launcher de RuneLite en la versión 2.6.3 o posterior.
2. Abrí su configuración: en Mac, `/Applications/RuneLite.app/Contents/MacOS/RuneLite --configure`.
3. En **client arguments** agregá `--insecure-write-credentials` y guardá.
4. Iniciá RuneLite **desde el Jagex Launcher** una vez: escribe `~/.runelite/credentials.properties`.
5. Corré `make plugin`: el cliente de desarrollo usa esas credenciales y entra con tu cuenta.

> **Cuidado.** `credentials.properties` permite entrar a tu cuenta **sin contraseña**: no lo compartas ni lo
> subas a ningún lado. Al terminar de probar, **borralo** (con eso vuelve el comportamiento normal) y, para
> invalidarlo, usá **End sessions** en la configuración de tu cuenta en runescape.com. Sacá también
> `--insecure-write-credentials` de los argumentos del launcher.

El plugin solo **lee** el estado del juego y solo cuando preguntás; no hay overlays, ni entradas de menú, ni
automatización (SPEC §1 y §10).

Con la sesión iniciada, el aviso gris de "Not logged in" ya no aparece. Preguntas que dependen del personaje:

| Escribí | Debería pasar |
|---|---|
| `¿Puedo hacer Monkey Madness II ahora? ¿Qué me falta?` | Lista requisitos cumplidos y faltantes **con tus niveles y quests reales**. Compará con tu personaje. |
| `¿Qué me conviene hacer para ganar GP con mis stats?` | Guías que **cumplís**, con precios del momento y su hora al pie. Si tu cuenta es ironman, **sin** GP por hora (no aplica) y lo dice. |
| `Tengo plata para flipear, ¿qué me conviene?` | Usa tu GP real (monedas del inventario, más el banco **si lo abriste** en esta sesión). |

**Criterio de aceptación (SPEC §13.7):** una pregunta que dependa de tu personaje real, respondida con
coherencia. Si el banco no se abrió, la respuesta debería decir que no lo conoce, no inventarlo.

## 5. Qué se manda y adónde

Al backend que configuraste (por defecto tu propia máquina): la pregunta, la foto del personaje (tipo de cuenta,
niveles, estado de quests, puntos de quest, inventario, GP, membresía si se puede saber, banco solo si lo
abriste) y tu key de OpenRouter en el header `X-OpenRouter-Key`. El backend usa la key solo durante ese pedido
y no la guarda ni la loguea. Detalle completo en `plugin/README.md`.
