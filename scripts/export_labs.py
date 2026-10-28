"""Pre-render de Quarto: exporta cada laboratorio Marimo (sesiones/*/lab.py) a HTML estático.

Se puede omitir con ``SKIP_LABS=1 quarto render`` (útil en vistas previas rápidas).
Los notebooks se ejecutan, así que el HTML contiene las salidas calculadas.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent


def main() -> int:
    if os.environ.get("SKIP_LABS"):
        print("SKIP_LABS definido: se omite la exportación de laboratorios.")
        return 0
    fallos = 0
    for lab in sorted((RAIZ / "sesiones").glob("s*/lab.py")):
        destino = lab.with_suffix(".html")
        if destino.exists() and destino.stat().st_mtime >= lab.stat().st_mtime:
            print(f"= {destino.relative_to(RAIZ)} al día")
            continue
        print(f"> exportando {lab.relative_to(RAIZ)}")
        r = subprocess.run(
            [sys.executable, "-m", "marimo", "export", "html", str(lab),
             "-o", str(destino), "--no-include-code", "--force"],
            cwd=RAIZ,
        )
        fallos += r.returncode != 0
    return 1 if fallos else 0


if __name__ == "__main__":
    raise SystemExit(main())
