"""The README explainer animation must stay consistent with the committed evidence."""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCENE = ROOT / "tools" / "animations" / "leakage_explainer.py"
GIF = ROOT / "docs" / "assets" / "leakage-explainer.gif"


def _scene_constants() -> dict[str, object]:
    """Read module-level literal assignments without importing manim."""
    values: dict[str, object] = {}
    for node in ast.parse(SCENE.read_text(encoding="utf-8")).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        try:
            value = ast.literal_eval(node.value)
        except ValueError:
            continue
        if isinstance(target, ast.Name):
            values[target.id] = value
        elif isinstance(target, ast.Tuple) and isinstance(value, tuple):
            for name, item in zip(target.elts, value, strict=True):
                if isinstance(name, ast.Name):
                    values[name.id] = item
    return values


def test_explainer_numbers_match_committed_evidence() -> None:
    metrics = json.loads(
        (ROOT / "reports" / "paired_a100" / "final_metrics.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (ROOT / "reports" / "protocol" / "paired_split_manifest.json").read_text(encoding="utf-8")
    )
    constants = _scene_constants()
    arms = metrics["aggregate"]["by_arm"]
    grouped = arms["grouped"]["map50"]
    leaky = arms["leaky_control"]["map50"]

    assert constants["GROUPED_MAP50"] == round(grouped["mean"], 4)
    assert constants["GROUPED_STD"] == round(grouped["std"], 4)
    assert constants["LEAKY_MAP50"] == round(leaky["mean"], 4)
    assert constants["LEAKY_STD"] == round(leaky["std"], 4)
    assert constants["DIFF_PP"] == round((leaky["mean"] - grouped["mean"]) * 100, 1)
    assert constants["BOARDS"] == sorted(
        {row["board_id"] for row in manifest["dataset"]["samples"]}
    )
    counts = manifest["counts"]
    assert counts["grouped_train"]["images"] == counts["leaky_train"]["images"]
    assert constants["TRAIN_TILES"] == round(counts["grouped_train"]["images"] / 10)
    assert constants["SIBLING_TILES"] == counts["leaky_exposure"]["images"] // 10
    assert counts["final_test"]["images"] == 30
    assert manifest["board_roles"]["final_test_and_exposure"] == "08"


def test_explainer_scene_stays_inside_the_public_boundary() -> None:
    source = SCENE.read_text(encoding="utf-8")

    assert "pcb_defect" not in source
    assert "data/pcb" not in source
    assert "ImageMobject" not in source
    assert "Tex(" not in source  # Text only: no LaTeX dependency


def test_readme_explainer_gif_is_small_synthetic_and_referenced() -> None:
    from PIL import Image

    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert GIF.is_file()
    assert GIF.stat().st_size <= 3_000_000
    with Image.open(GIF) as gif:
        assert gif.format == "GIF"
        assert gif.size == (640, 360)
        assert getattr(gif, "is_animated", False)
    assert "docs/assets/leakage-explainer.gif" in readme
    assert "不含任何 dataset 影像" in readme
    assert "tools/animations/README.md" in readme
