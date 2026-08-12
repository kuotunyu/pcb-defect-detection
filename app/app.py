"""Entry point for the evidence-first PCB review workstation."""

from __future__ import annotations

from app.theme import APP_CSS, APP_THEME
from app.ui import create_demo

demo = create_demo()


if __name__ == "__main__":
    demo.launch(
        theme=APP_THEME,
        css=APP_CSS,
        footer_links=[],
        show_error=True,
    )
