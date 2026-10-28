#!/bin/bash
# OSRS Agent launcher: double-click it (or `make play`).
#
# Starts what the plugin needs, in order -- Postgres, the backend, the RuneLite dev client with the
# plugin -- and stops what IT started when the client closes, on Ctrl+C or when this window closes.
# Anything that was already running (your own `make backend`, a Postgres you started) is reused and
# never stopped. A second launcher, or a second dev client, is refused.
#
# Test knobs (environment): PG_BIN PG_DATA PG_LOG PG_PORT BACKEND_PORT BACKEND_CMD CLIENT_CMD
# OSRS_AGENT_STATE.

set -u

SELF="$0"
while [ -L "$SELF" ]; do
	target="$(readlink "$SELF")"
	case "$target" in /*) SELF="$target" ;; *) SELF="$(dirname "$SELF")/$target" ;; esac
done
SELF_NAME="$(basename "$SELF")"
REPO="${OSRS_AGENT_REPO:-$(cd "$(dirname "$SELF")/.." && pwd)}"

export PATH="/opt/homebrew/bin:/opt/homebrew/opt/postgresql@17/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if [ -z "${JAVA_HOME:-}" ] && [ -x /usr/libexec/java_home ]; then
	JAVA_HOME="$(/usr/libexec/java_home 2>/dev/null)" && export JAVA_HOME
fi

PG_PREFIX="/opt/homebrew"
PG_BIN="${PG_BIN:-$PG_PREFIX/opt/postgresql@17/bin}"
PG_DATA="${PG_DATA:-$PG_PREFIX/var/postgresql@17}"
PG_LOG="${PG_LOG:-$PG_PREFIX/var/log/postgresql@17.log}"
PG_PORT="${PG_PORT:-5432}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
DEFAULT_BACKEND_CMD="$REPO/backend/.venv/bin/python -m app.main"
BACKEND_CMD="${BACKEND_CMD:-$DEFAULT_BACKEND_CMD}"
CLIENT_CMD="${CLIENT_CMD:-./gradlew run --no-daemon --console=plain -q}"
STATE="${OSRS_AGENT_STATE:-${XDG_STATE_HOME:-$HOME/.local/state}/osrs-agent}"
LOGS="$HOME/Library/Logs/osrs-agent"
LOCK="$STATE/lock"
LEFTOVERS="$STATE/started"          # "<kind> <pid>" for everything this launcher started
CLIENT_PATTERN="com.osrsagent.AgentPluginTest"
BACKEND_WAIT_S="${BACKEND_WAIT_S:-180}"

mkdir -p "$STATE" "$LOGS"

say() { printf '%s\n' "$*"; }
die()
{
	say ""
	say "✗ $*"
	if [ -t 0 ]; then
		printf 'Presioná Enter para cerrar... '
		read -r _ || true
	fi
	exit 1
}

alive() { kill -0 "$1" 2>/dev/null; }
cmd_of() { ps -p "$1" -o command= 2>/dev/null; }
# Remember what we started, with its exact command line, so a later run can tell it is still the same
# process (and not a recycled PID) before touching it.
record()
{
	local i
	# a freshly forked subshell still shows this script's command line until it execs
	for i in $(seq 1 30); do
		[ "$(cmd_of "$2")" != "$(cmd_of $$)" ] && break
		sleep 0.1
	done
	printf '%s|%s|%s\n' "$1" "$2" "$(cmd_of "$2")" >> "$LEFTOVERS"
}

descendants()
{
	local child
	for child in $(pgrep -P "$1" 2>/dev/null); do
		say "$child"
		descendants "$child"
	done
}

# TERM the process and everything below it, wait, then KILL what is left.
stop_tree()
{
	local pids i p any
	pids="$1 $(descendants "$1" | tr '\n' ' ')"
	for p in $pids; do kill -TERM "$p" 2>/dev/null; done
	for i in $(seq 1 "${2:-15}"); do
		any=0
		for p in $pids; do alive "$p" && any=1; done
		[ "$any" = 0 ] && return 0
		sleep 1
	done
	for p in $pids; do kill -KILL "$p" 2>/dev/null; done
}

pg_up() { "$PG_BIN/pg_isready" -h 127.0.0.1 -p "$PG_PORT" -q; }
backend_up() { curl -fsS -m 3 "http://127.0.0.1:$BACKEND_PORT/healthz" >/dev/null 2>&1; }
listener_on() { lsof -nP -iTCP:"$1" -sTCP:LISTEN -t 2>/dev/null | head -1; }

PG_STARTED=0
BACKEND_PID=""
CLIENT_PID=""
CLEANED=0

cleanup()
{
	[ "$CLEANED" = 1 ] && return
	CLEANED=1
	trap '' INT TERM HUP
	say ""
	if [ -n "$CLIENT_PID" ] && alive "$CLIENT_PID"; then
		say "Cerrando el cliente de RuneLite..."
		stop_tree "$CLIENT_PID" 20
	fi
	if [ -n "$BACKEND_PID" ]; then
		if alive "$BACKEND_PID"; then
			say "Apagando el backend que arranqué (PID $BACKEND_PID)..."
			stop_tree "$BACKEND_PID" 10
		fi
	fi
	if [ "$PG_STARTED" = 1 ]; then
		say "Apagando el Postgres que arranqué..."
		"$PG_BIN/pg_ctl" -D "$PG_DATA" -m fast -w stop >/dev/null 2>&1
	fi
	rm -f "$LEFTOVERS"
	rm -rf "$LOCK"
	say "Listo: no quedó nada mío corriendo. Podés cerrar esta ventana."
}

# A previous launcher that was killed hard (kill -9, power loss) leaves its children behind: stop
# those, and only those, after checking the PID still belongs to what we started.
recover_leftovers()
{
	local kind pid cmd
	[ -f "$LEFTOVERS" ] || return 0
	while IFS='|' read -r kind pid cmd; do
		[ -n "${pid:-}" ] && alive "$pid" && [ "$(cmd_of "$pid")" = "$cmd" ] || continue
		say "Quedó un $kind de una sesión anterior (PID $pid): lo apago."
		if [ "$kind" = postgres ]; then
			"$PG_BIN/pg_ctl" -D "$PG_DATA" -m fast -w stop >/dev/null 2>&1
		else
			stop_tree "$pid" 15
		fi
	done < "$LEFTOVERS"
	rm -f "$LEFTOVERS"
}

acquire_lock()
{
	local old
	if mkdir "$LOCK" 2>/dev/null; then
		say $$ > "$LOCK/pid"
		return 0
	fi
	old="$(cat "$LOCK/pid" 2>/dev/null)"
	if [ -n "$old" ] && alive "$old" && cmd_of "$old" | grep -qF "$SELF_NAME"; then
		die "Ya hay un lanzador corriendo (PID $old). Usá esa ventana; si no la ves, cerrá el cliente de RuneLite y se apagará solo."
	fi
	recover_leftovers
	rm -rf "$LOCK"
	mkdir "$LOCK" 2>/dev/null || die "No pude crear el candado en $LOCK"
	say $$ > "$LOCK/pid"
}

say "OSRS Agent — lanzador"
acquire_lock
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

other="$(pgrep -f "$CLIENT_PATTERN" | head -1)"
[ -z "$other" ] || die "Ya hay un cliente de desarrollo de RuneLite abierto (PID $other). Cerralo antes de lanzar otro."
if [ "$BACKEND_CMD" = "$DEFAULT_BACKEND_CMD" ] && [ ! -x "$REPO/backend/.venv/bin/python" ]; then
	die "Falta el entorno del backend. Corré una vez: cd backend && uv sync"
fi
command -v java >/dev/null 2>&1 || [ -n "${JAVA_HOME:-}" ] || die "No encuentro Java 11+ (JAVA_HOME)."

# 1. Postgres ----------------------------------------------------------------------------------
if pg_up; then
	say "• Postgres: ya estaba corriendo, lo reuso (no lo voy a apagar)."
else
	say "• Postgres: arrancando..."
	"$PG_BIN/pg_ctl" -D "$PG_DATA" -l "$PG_LOG" -o "-p $PG_PORT" -w start >/dev/null 2>&1 || die "Postgres no arrancó. Mirá $PG_LOG"
	PG_STARTED=1
	record postgres "$(head -1 "$PG_DATA/postmaster.pid" 2>/dev/null)"
	pg_up || die "Postgres arrancó pero no acepta conexiones."
fi

# 2. Backend -----------------------------------------------------------------------------------
listener="$(listener_on "$BACKEND_PORT")"
if [ -n "$listener" ]; then
	if backend_up; then
		say "• Backend: ya había uno respondiendo en el puerto $BACKEND_PORT (PID $listener), lo reuso (no lo voy a apagar)."
		say "  Si lo arrancaste antes de cambiar el código, cerralo y volvé a lanzar para usar la versión nueva."
	else
		die "El puerto $BACKEND_PORT lo usa otro programa (PID $listener) y no responde como el backend."
	fi
else
	say "• Backend: arrancando (la primera vez tarda en cargar el modelo)..."
	( cd "$REPO/backend" && PORT="$BACKEND_PORT" exec $BACKEND_CMD ) > "$LOGS/backend.log" 2>&1 &
	BACKEND_PID=$!
	record backend "$BACKEND_PID"
	waited=0
	until backend_up; do
		alive "$BACKEND_PID" || { tail -n 15 "$LOGS/backend.log"; die "El backend se cayó al arrancar (log: $LOGS/backend.log)."; }
		[ "$waited" -lt "$BACKEND_WAIT_S" ] || die "El backend no respondió en $BACKEND_WAIT_S s (log: $LOGS/backend.log)."
		sleep 1
		waited=$((waited + 1))
	done
	say "  listo en $waited s (PID $BACKEND_PID, log: $LOGS/backend.log)"
fi

# 3. RuneLite dev client ----------------------------------------------------------------------
say "• Cliente de RuneLite: abriendo (tarda unos segundos)..."
started_at=$SECONDS
( cd "$REPO/plugin" && exec $CLIENT_CMD ) > "$LOGS/client.log" 2>&1 &
CLIENT_PID=$!
record client "$CLIENT_PID"
say ""
say "Todo en marcha. Iniciá sesión desde la ventana de RuneLite que se abrió."
say "Para terminar: cerrá RuneLite (o esta ventana, o Ctrl+C) y se apaga lo que arranqué."

warned=0
while alive "$CLIENT_PID"; do
	sleep 2 &
	wait $! 2>/dev/null
	if [ -n "$BACKEND_PID" ] && [ "$warned" = 0 ] && ! alive "$BACKEND_PID"; then
		warned=1
		say "! El backend se cayó (log: $LOGS/backend.log). Cerrá RuneLite y volvé a lanzar."
	fi
done
wait "$CLIENT_PID" 2>/dev/null
status=$?
if [ $((SECONDS - started_at)) -lt 30 ] && [ "$status" -ne 0 ]; then
	say "! El cliente terminó enseguida (código $status). Últimas líneas de $LOGS/client.log:"
	tail -n 15 "$LOGS/client.log"
fi
exit 0
