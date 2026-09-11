"""README explainer for the paired board-leakage experiment (Manim Community).

Every number shown comes from the committed evidence in
``reports/paired_a100/final_metrics.json`` and ``reports/protocol/paired_split_manifest.json``;
``tests/test_animations.py`` keeps the constants below in sync with those files. Only abstract
shapes are drawn: the animation contains no dataset image.

Preview:  manim render -ql leakage_explainer.py LeakageExplainer
Final:    manim render -qm leakage_explainer.py LeakageExplainer
See README.md in this directory for the environment, font, and GIF steps.
"""

from __future__ import annotations

from manim import (
    DOWN,
    ORIGIN,
    RIGHT,
    UP,
    Arrow,
    Brace,
    Create,
    DashedVMobject,
    FadeIn,
    FadeOut,
    GrowFromEdge,
    LaggedStart,
    Line,
    Mobject,
    Rectangle,
    RoundedRectangle,
    Scene,
    Square,
    SurroundingRectangle,
    Text,
    VGroup,
    config,
)

# Noto Sans TC drops Latin word spaces in Manim's Text pipeline; JhengHei keeps them.
FONT = "Microsoft JhengHei"
BG = "#0f1116"
PANEL = "#1b1f2a"
TRAIN = "#4f8ef7"
BOARD08 = "#f5a623"
DIM = "#3a3f4b"
GREY_TEXT = "#9aa3b2"
HILITE = "#ffd166"
WHITE = "#f2f4f8"
LINE = "#4a5160"

# Committed evidence (final_metrics.json / paired_split_manifest.json)
GROUPED_MAP50, GROUPED_STD = 0.6330, 0.1491
LEAKY_MAP50, LEAKY_STD = 0.8456, 0.0375
DIFF_PP = 21.3
BOARDS = ["01", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
TILES_PER_BOARD = 6  # each tile stands for roughly 10 images
TRAIN_TILES = 51  # 513 training images / 10
SIBLING_TILES = 3  # 30 Board 08 sibling images / 10

config.background_color = BG


def T(text: str, size: int = 28, color: str = WHITE) -> Text:
    return Text(text, font=FONT, font_size=size, color=color, line_spacing=0.9)


def board(board_id: str) -> VGroup:
    rect = RoundedRectangle(
        corner_radius=0.15,
        width=2.0,
        height=1.4,
        stroke_color=LINE,
        stroke_width=2,
        fill_color=PANEL,
        fill_opacity=1,
    )
    label = T(f"Board {board_id}", 18, GREY_TEXT).move_to(rect.get_top() + DOWN * 0.24)
    tiles = VGroup(
        *[
            Square(0.3, fill_color=TRAIN, fill_opacity=0.95, stroke_width=0)
            for _ in range(TILES_PER_BOARD)
        ]
    )
    tiles.arrange_in_grid(rows=2, cols=3, buff=0.1).move_to(rect.get_center() + DOWN * 0.15)
    group = VGroup(rect, label, tiles)
    group.rect, group.label, group.tiles = rect, label, tiles
    return group


def tile_block(n: int, color: str, cols: int = 17, side: float = 0.22) -> VGroup:
    block = VGroup(
        *[Square(side, fill_color=color, fill_opacity=0.95, stroke_width=0) for _ in range(n)]
    )
    return block.arrange_in_grid(cols=cols, buff=0.06)


def panel(width: float, height: float, dashed: bool = False, color: str = LINE):
    rect = RoundedRectangle(
        corner_radius=0.18,
        width=width,
        height=height,
        stroke_color=color,
        stroke_width=2.5,
        fill_color=PANEL,
        fill_opacity=0.6,
    )
    return DashedVMobject(rect, num_dashes=40) if dashed else rect


class LeakageExplainer(Scene):
    def fade_others(self, keep: list[Mobject]) -> FadeOut:
        """Fade every top-level mobject except the given ones and their families.

        Animating a submobject makes Manim split its parent group into top-level pieces,
        so fading the original group object later misses them; this catches whatever is
        actually on screen.
        """
        keep_ids = {id(m) for k in keep for m in k.get_family()}
        return FadeOut(*[m for m in self.mobjects if id(m) not in keep_ids])

    def construct(self) -> None:
        # ---------- Beat 1: dataset and boards ----------
        title = T("Split 方式如何改變 PCB 瑕疵偵測的成績", 40).to_edge(UP, buff=0.5)
        subtitle = T("Paired board-leakage experiment · 3 seeds", 22, GREY_TEXT).next_to(
            title, DOWN, buff=0.15
        )
        boards = [board(b) for b in BOARDS]
        grid = VGroup(*boards).arrange_in_grid(rows=2, cols=5, buff=0.35)
        grid.move_to(ORIGIN + DOWN * 0.15)
        caption = T("HRIPCB：693 張影像、10 塊 template board；每格約 10 張", 26, GREY_TEXT)
        caption.to_edge(DOWN, buff=0.45)

        self.play(FadeIn(title, shift=DOWN * 0.3), FadeIn(subtitle), run_time=1.2)
        self.play(
            LaggedStart(*[FadeIn(b, scale=0.9) for b in boards], lag_ratio=0.08), run_time=2.2
        )
        self.play(FadeIn(caption), run_time=0.8)
        self.wait(1.0)

        b08 = boards[BOARDS.index("08")]
        b01 = boards[BOARDS.index("01")]
        hilite = SurroundingRectangle(b08, color=HILITE, buff=0.08, stroke_width=3)
        caption2 = T("同一塊 board 的影像高度相似（siblings），彼此不是獨立樣本", 26, HILITE)
        caption2.to_edge(DOWN, buff=0.45)
        self.play(Create(hilite), FadeOut(caption), FadeIn(caption2), run_time=1.0)
        self.play(b08.tiles.animate.set_fill(BOARD08), run_time=1.0)
        self.wait(1.2)

        # ---------- Beat 2: common final test and sibling pool ----------
        self.play(
            FadeOut(title), FadeOut(subtitle), FadeOut(caption2), FadeOut(hilite), run_time=0.8
        )
        self.play(grid.animate.scale(0.8).move_to([-2.1, 0.9, 0]), run_time=1.0)

        test_panel = panel(3.6, 1.7, dashed=True, color=BOARD08).move_to([4.9, 1.7, 0])
        test_title = T("共同 final test", 22, BOARD08)
        test_title.next_to(test_panel.get_top(), DOWN, buff=0.15)
        test_note = T("Board 08 · 30 張 · 鎖定，兩組共用", 16, GREY_TEXT)
        test_note.next_to(test_panel.get_bottom(), UP, buff=0.15)
        sib_panel = panel(3.2, 1.3, color=BOARD08).move_to([4.9, -0.6, 0])
        sib_title = T("sibling pool", 22, BOARD08).next_to(sib_panel.get_top(), DOWN, buff=0.12)
        sib_note = T("Board 08 · 另外 30 張", 16, GREY_TEXT)
        sib_note.next_to(sib_panel.get_bottom(), UP, buff=0.12)
        self.play(
            Create(test_panel),
            FadeIn(test_title),
            FadeIn(test_note),
            Create(sib_panel),
            FadeIn(sib_title),
            FadeIn(sib_note),
            run_time=1.2,
        )

        # Fly copies of the Board 08 tiles so the board group itself is never re-parented.
        test_tiles = VGroup(*[t.copy() for t in b08.tiles[:3]])
        sib_tiles = VGroup(*[t.copy() for t in b08.tiles[3:]])
        self.add(test_tiles, sib_tiles)
        self.play(
            FadeOut(b08.tiles),
            test_tiles.animate.arrange(RIGHT, buff=0.15)
            .scale(1.25)
            .move_to(test_panel.get_center() + DOWN * 0.05),
            sib_tiles.animate.arrange(RIGHT, buff=0.15).scale(1.25).move_to(sib_panel.get_center()),
            run_time=1.6,
        )
        b01_note = T("validation / calibration", 14, GREY_TEXT).next_to(b01, DOWN, buff=0.08)
        self.play(
            b01.tiles.animate.set_fill(DIM),
            b01.rect.animate.set_stroke(DIM),
            FadeIn(b01_note),
            run_time=0.8,
        )
        caption3 = T(
            "Final test 固定為 Board 08 的 30 張；剩下 30 張 siblings 先放在旁邊", 26, GREY_TEXT
        )
        caption3.to_edge(DOWN, buff=0.45)
        self.play(FadeIn(caption3), run_time=0.8)
        self.wait(1.6)

        # ---------- Beat 3: two arms ----------
        keep_test = VGroup(test_panel, test_title, test_note, test_tiles)
        keep_sib = VGroup(sib_panel, sib_title, sib_note, sib_tiles)
        self.play(self.fade_others([keep_test, keep_sib]), run_time=0.9)
        self.play(
            keep_test.animate.scale(0.85).move_to([-1.7, 2.6, 0]),
            keep_sib.animate.scale(0.85).move_to([2.6, 2.6, 0]),
            run_time=1.0,
        )

        g_head = T("Grouped", 30, TRAIN).move_to([-3.5, 1.15, 0])
        g_block = tile_block(TRAIN_TILES, TRAIN).move_to([-3.5, 0.0, 0])
        g_box = SurroundingRectangle(
            g_block, color=TRAIN, buff=0.18, stroke_width=2.5, corner_radius=0.12
        )
        g_note = T("train 513 張 · Boards 04–12\nBoard 08：0 張", 18, GREY_TEXT)
        g_note.move_to([-3.5, -1.3, 0])
        self.play(
            FadeIn(g_head),
            FadeIn(g_box),
            FadeIn(g_block, lag_ratio=0.01),
            FadeIn(g_note),
            run_time=1.4,
        )

        l_head = T("Leaky control", 30, BOARD08).move_to([3.5, 1.15, 0])
        l_block = tile_block(TRAIN_TILES, TRAIN).move_to([3.5, 0.0, 0])
        l_box = SurroundingRectangle(
            l_block, color=BOARD08, buff=0.18, stroke_width=2.5, corner_radius=0.12
        )
        self.play(FadeIn(l_head), FadeIn(l_box), FadeIn(l_block, lag_ratio=0.01), run_time=1.2)

        swapped_out = VGroup(*l_block[-SIBLING_TILES:])
        incoming = VGroup(*[t.copy() for t in sib_tiles])
        targets = [t.copy().set_fill(BOARD08) for t in swapped_out]
        self.play(
            FadeOut(swapped_out, shift=DOWN * 0.4),
            *[inc.animate.become(tgt) for inc, tgt in zip(incoming, targets, strict=True)],
            run_time=1.6,
        )
        l_note = T("train 513 張 · 483 張其他 boards\n+ 30 張 Board 08 siblings", 18, GREY_TEXT)
        l_note.move_to([3.5, -1.3, 0])
        caption4 = T(
            "唯一差別：train 是否看過 Board 08 siblings\n"
            "train size、class 分布、base model、recipe、seeds 42／43／44 全部相同",
            22,
            GREY_TEXT,
        ).move_to([0, -3.05, 0])
        self.play(FadeIn(l_note), FadeIn(caption4), run_time=0.9)
        self.wait(2.2)

        # ---------- Beat 4: same test, two results ----------
        target = keep_test.get_bottom() + DOWN * 0.05
        arrow_g = Arrow(g_box.get_top(), target, buff=0.1, color=TRAIN, stroke_width=3)
        arrow_l = Arrow(l_box.get_top(), target, buff=0.1, color=BOARD08, stroke_width=3)
        self.play(GrowFromEdge(arrow_g, DOWN), GrowFromEdge(arrow_l, DOWN), run_time=0.8)
        self.wait(0.8)
        self.play(
            self.fade_others([g_head, l_head, keep_test]),
            keep_test.animate.move_to([0, 2.6, 0]),
            run_time=0.9,
        )

        base_y = -2.4
        scale = 3.8
        baseline = Line([-4.4, base_y, 0], [3.6, base_y, 0], color=LINE, stroke_width=2)
        axis_label = T("mAP50 · 3 seeds 平均 ± std", 20, GREY_TEXT).move_to([-0.4, -3.2, 0])

        def bar(x: float, value: float, color: str) -> Rectangle:
            rect = Rectangle(
                width=1.4, height=value * scale, fill_color=color, fill_opacity=0.9, stroke_width=0
            )
            rect.move_to([x, base_y + value * scale / 2, 0])
            return rect

        g_bar = bar(-2.4, GROUPED_MAP50, TRAIN)
        l_bar = bar(1.4, LEAKY_MAP50, BOARD08)
        self.play(
            g_head.animate.move_to([-2.4, base_y - 0.35, 0]),
            l_head.animate.move_to([1.4, base_y - 0.35, 0]),
            Create(baseline),
            FadeIn(axis_label),
            run_time=0.9,
        )
        g_val = T(f"{GROUPED_MAP50:.4f} ± {GROUPED_STD:.4f}", 24, TRAIN)
        l_val = T(f"{LEAKY_MAP50:.4f} ± {LEAKY_STD:.4f}", 24, BOARD08)
        self.play(GrowFromEdge(g_bar, DOWN), run_time=1.2)
        self.play(FadeIn(g_val.next_to(g_bar, UP, buff=0.15)), run_time=0.5)
        self.play(GrowFromEdge(l_bar, DOWN), run_time=1.2)
        self.play(FadeIn(l_val.next_to(l_bar, UP, buff=0.15)), run_time=0.5)

        gap = Line([2.35, g_bar.get_top()[1], 0], [2.35, l_bar.get_top()[1], 0])
        brace = Brace(gap, direction=RIGHT, color=HILITE)
        gap_text = T(f"+{DIFF_PP} pp", 30, HILITE)
        brace.put_at_tip(gap_text)
        dotted = DashedVMobject(
            Line(
                [-1.7, g_bar.get_top()[1], 0],
                [2.35, g_bar.get_top()[1], 0],
                color=TRAIN,
                stroke_width=2,
            ),
            num_dashes=24,
        )
        self.play(Create(dotted), GrowFromEdge(brace, DOWN), FadeIn(gap_text), run_time=1.0)
        caption5 = T(
            "看過 30 張 sibling images，同一個模型 recipe 的 mAP50 高了 21.3 個百分點", 24, HILITE
        )
        caption5.move_to([0, -3.7, 0])
        self.play(FadeIn(caption5), run_time=0.8)
        self.wait(2.4)

        # ---------- Beat 5: takeaway ----------
        self.play(self.fade_others([]), run_time=0.9)
        line1 = T("Split policy 比換模型更重要", 46).move_to(UP * 0.8)
        line2 = T(
            "同一個 recipe，只差 30 張 Board 08 siblings，mAP50 相差 21.3 個百分點", 26, GREY_TEXT
        )
        line2.next_to(line1, DOWN, buff=0.4)
        line3 = T("Final test 只有一塊 board、30 張影像；不推論產線泛化", 20, GREY_TEXT)
        line3.next_to(line2, DOWN, buff=0.9)
        line4 = T("github.com/kuotunyu/pcb-defect-detection", 20, "#6f7a8c")
        line4.next_to(line3, DOWN, buff=0.2)
        self.play(FadeIn(line1, shift=UP * 0.2), run_time=1.0)
        self.play(FadeIn(line2), run_time=0.8)
        self.play(FadeIn(line3), FadeIn(line4), run_time=0.8)
        self.wait(2.5)
