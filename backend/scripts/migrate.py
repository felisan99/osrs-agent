"""Apply pending SQL migrations from ``supabase/migrations`` to ``SUPABASE_DB_URL``.

uv run python -m scripts.migrate
"""

from __future__ import annotations

import sys

from app.config import get_settings
from app.db import MIGRATIONS_DIR, MissingDatabaseUrl, apply_migrations, require_db_url


def main() -> int:
    try:
        url = require_db_url(get_settings())
    except MissingDatabaseUrl as exc:
        print(exc, file=sys.stderr)
        return 2
    applied = apply_migrations(url)
    print(f"migrations dir: {MIGRATIONS_DIR}")
    print("applied:", ", ".join(applied) if applied else "nothing (already up to date)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
