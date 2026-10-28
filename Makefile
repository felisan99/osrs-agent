# Atajos del repo (macOS + Homebrew). `make` sin argumentos lista los comandos.
# Postgres: Homebrew 17 con pgvector, arrancado con pg_ctl (no con brew services).

PG_PREFIX ?= /opt/homebrew
PG_BIN := $(PG_PREFIX)/opt/postgresql@17/bin
PG_DATA := $(PG_PREFIX)/var/postgresql@17
PG_LOG := $(PG_PREFIX)/var/log/postgresql@17.log

.DEFAULT_GOAL := help
.PHONY: help db backend plugin play test test-backend test-plugin retrieval retrieval-heldout eval memory

help:
	@echo "make db            arranca Postgres si no esta corriendo"
	@echo "make backend       arranca Postgres y la API en http://127.0.0.1:8000 (Ctrl+C para parar)"
	@echo "make plugin        abre el cliente de RuneLite en modo desarrollador con el plugin"
	@echo "make play          todo junto (Postgres, backend, cliente) y lo apaga al cerrar; ver docs/lanzador.md"
	@echo "make test          TODOS los tests (backend y plugin): una vez por tanda de cambios, no por edicion"
	@echo "make test-backend  solo el backend;  make test-plugin  solo el plugin"
	@echo "make retrieval     benchmark de retrieval, set de desarrollo (segundos, gratis); ver docs/rag-quality.md"
	@echo "make retrieval-heldout   lo mismo con el set reservado (preguntas que no se usaron para ajustar nada)"
	@echo "make eval          evaluacion end-to-end (gasta ~USD 0.03 de la key de desarrollo en .env)"
	@echo "make memory        estado de la memoria de conversaciones (hilos y checkpoints)"
	@echo "Guia paso a paso para probar el plugin: docs/probar-el-plugin.md"

db:
	@$(PG_BIN)/pg_isready -h 127.0.0.1 -p 5432 -q && echo "Postgres ya esta corriendo" || \
		($(PG_BIN)/pg_ctl -D $(PG_DATA) -l $(PG_LOG) start && sleep 2 && $(PG_BIN)/pg_isready -h 127.0.0.1 -p 5432)

backend: db
	cd backend && uv run python -m app.main

plugin:
	cd plugin && ./gradlew run

play:
	@launcher/osrs-agent.command

test: test-backend test-plugin

test-backend: db
	cd backend && uv run pytest -q

test-plugin:
	cd plugin && ./gradlew test

retrieval: db
	cd backend && uv run python -m eval.retrieval_bench

retrieval-heldout: db
	cd backend && uv run python -m eval.retrieval_bench --cases eval/retrieval_heldout.yaml

eval: db
	cd backend && uv run python -m eval.run_eval

memory: db
	cd backend && uv run python -m scripts.maintenance stats
