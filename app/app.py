"""Entry point for the evidence-first PCB review workstation."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.theme import APP_CSS, APP_HEAD, APP_JS, APP_THEME
from app.ui import create_demo

demo = create_demo()


if __name__ == "__main__" and "--check" in sys.argv:
    print(demo.title)
elif __name__ == "__main__":
    demo.launch(
        theme=APP_THEME,
        css=APP_CSS,
        head=APP_HEAD,
        js=APP_JS,
        footer_links=[],
        show_error=True,
    )
