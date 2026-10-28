# Lanzador de un clic

`launcher/osrs-agent.command` arranca todo lo que necesita el plugin y lo apaga al terminar. Se usa con doble clic
en Finder (se abre una ventana de Terminal con el estado) o con `make play`. Podés arrastrarlo al Dock o hacerle
un alias.

Primera vez: `cd backend && uv sync` (crea el entorno del backend) y, si macOS pregunta por permisos al abrir
el archivo, clic derecho > Abrir.

## Qué hace, en orden

1. **Postgres:** si ya corre, lo reusa; si no, lo arranca.
2. **Backend** (`127.0.0.1:8000`): si ya hay uno sano en ese puerto, lo reusa; si el puerto lo usa otro programa,
   avisa y para; si no hay nada, lo arranca (tarda unos segundos en cargar el modelo). Log en
   `~/Library/Logs/osrs-agent/backend.log`.
3. **Cliente de RuneLite en modo desarrollador** con el plugin (`./gradlew run --no-daemon`, sin dejar el daemon
   de Gradle en segundo plano). Log en `~/Library/Logs/osrs-agent/client.log`.

## Que no se duplique

* Un **segundo lanzador** se niega a arrancar y dice el PID del primero.
* Si ya hay un **cliente de desarrollo** abierto, se niega antes de tocar nada.
* Un backend o un Postgres que ya estaban corriendo **se reusan**, y se anota que no son suyos.

## Qué se apaga al cerrar

Se apaga **solo lo que el lanzador arrancó**, en este orden: cliente, backend, Postgres. Lo que ya estaba
corriendo antes (tu `make backend`, un Postgres que dejaste abierto) no se toca.

| Cómo termina | Resultado |
|---|---|
| Cerrás la ventana de RuneLite (lo normal) | apaga backend y Postgres si eran suyos y sale |
| Ctrl+C en la ventana del lanzador | cierra el cliente y apaga lo suyo |
| Cerrás la ventana de Terminal | igual (recibe la señal de cierre) |
| El lanzador muere de golpe (`kill -9`, corte de luz) | en la **próxima** ejecución detecta lo que quedó y lo apaga, solo si el proceso sigue siendo el mismo (compara el comando exacto) |

**Cerrá RuneLite normalmente antes que la ventana de Terminal.** Si cerrás la Terminal con la sesión del juego
abierta, el cliente muere de golpe y tu personaje puede quedar conectado en el juego un minuto, sin que lo controles.

El **Jagex Launcher** no interviene: solo se usó una vez para generar `credentials.properties`. Cerrarlo no
afecta a nada de esto.

## Límites

* Es el cliente de desarrollo, no tu RuneLite de siempre: el lanzador oficial ignora los plugins sueltos y la
  única vía para eso es el Plugin Hub (ver README).
* Si ya tenías el backend corriendo desde antes de cambiar el código, se reusa **esa** versión: cerralo para
  que el lanzador arranque uno nuevo.
* Una `credentials.properties` en `~/.runelite` es lo que permite entrar sin contraseña; al terminar de
  probar, borrala, usá "End sessions" en runescape.com y quitá `--insecure-write-credentials` del lanzador.

## Cómo se probó

Con procesos de mentira para el cliente, el backend y un Postgres temporal (arrancar y apagar lo propio,
reusar lo ajeno, cierre por cierre natural, TERM, HUP y Ctrl+C, segundo lanzador, cliente ya abierto, muerte con
`kill -9` y recuperación) y una corrida real con Gradle y el cliente de RuneLite aislado: cargó el plugin, y al
cerrar no quedó ningún proceso de Java; los archivos reales de `~/.runelite` no cambiaron.
