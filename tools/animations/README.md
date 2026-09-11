# README explainer animation

`leakage_explainer.py` renders the 42-second explainer embedded in the top-level README
(`docs/assets/leakage-explainer.gif`). It draws only abstract shapes and the committed aggregate
numbers; it contains no dataset image, so it sits inside the public release boundary.
`tests/test_animations.py` fails if the numbers in the scene drift from
`reports/paired_a100/final_metrics.json` or `reports/protocol/paired_split_manifest.json`.

Manim is deliberately **not** part of the locked project environment or CI. Render it in a
throwaway environment:

```bash
uv venv --python 3.11 .venv-manim
uv pip install --python .venv-manim manim==0.21.0
```

Requirements observed on 2026-09-11: Manim Community 0.21.0 (bundles PyAV, so no system ffmpeg is
needed for the MP4), and a CJK font. The scene uses **Microsoft JhengHei**. Noto Sans TC renders
Traditional glyphs but drops the spaces between Latin words in Manim's `Text` pipeline
(`3 seeds` becomes `3seeds`), so keep JhengHei on Windows; on Linux install a CJK font and change
`FONT` at the top of the scene.

```bash
cd tools/animations
# quick preview, 854x480 at 15 fps
manim render -ql --media_dir media leakage_explainer.py LeakageExplainer
# final MP4, 1280x720 at 30 fps
manim render -qm --media_dir media -o leakage_720p leakage_explainer.py LeakageExplainer
```

The README GIF is produced from the MP4 with a two-pass palette so it stays under 3 MB:

```bash
ffmpeg -i media/videos/leakage_explainer/720p30/leakage_720p.mp4 \
  -vf "fps=12,scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=96:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle" \
  ../../docs/assets/leakage-explainer.gif
```

`media/` and `out/` in this directory are git-ignored render outputs.

Implementation note: Manim re-parents a group's children into top-level scene objects as soon as
one child is animated, so a later `FadeOut(group)` misses them. `LeakageExplainer.fade_others`
fades whatever is actually on screen except an explicit keep list; use it instead of fading the
original group objects.
