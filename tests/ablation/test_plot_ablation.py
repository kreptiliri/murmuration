import json

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
plot_ablation = pytest.importorskip("plot_ablation")

RULES = ("separation", "alignment", "cohesion")
BATCHES = [0, 1, 2, 4, 8, 100, 1000, 40950]


def fake_curve():
    #made-up numbers that differ by rule, block and batch, so a mix-up shows
    curve = []
    for i, batch in enumerate(BATCHES):
        done, blocks = {}, []
        for r, rule in enumerate(RULES):
            done[rule] = round(0.1 * i + 0.01 * r, 4)
        for k in range(2):
            lost = {}
            for r, rule in enumerate(RULES):
                lost[rule] = round(0.05 * i + 0.02 * r + 0.3 * k, 4)
            blocks.append({"block": k, "score": 0.5, "lost": lost, "explained": 0.9})
        curve.append({"batch": batch, "pass": batch / 1365, "normal": 0.9, "done": done, "blocks": blocks})
    return {"curve": curve, "check_clips": 20}


def fake_readability():
    curve = []
    for i, batch in enumerate(BATCHES):
        scores = {}
        for r, rule in enumerate(RULES):
            scores[rule] = [0.0, round(0.4 + 0.01 * i + 0.1 * r, 4), round(0.5 + 0.01 * i, 4)]
        curve.append({"batch": batch, "pass": batch / 1365, "scores": scores})
    return {"curve": curve, "learn_clips": 50, "check_clips": 20}


def lines_labelled(ax, word):
    return [line for line in ax.get_lines() if word in line.get_label()]


def test_one_panel_per_rule():
    fig = plot_ablation.plot_ablation(fake_curve())
    axes = fig.get_axes()
    assert len(axes) == 3
    for ax, rule in zip(axes, RULES):
        assert rule in ax.get_title(loc="left").lower()


def test_what_the_model_does_and_what_each_round_carries_are_drawn():
    data = fake_curve()
    fig = plot_ablation.plot_ablation(data)
    rows = [row for row in data["curve"] if row["batch"] > 0]
    for ax, rule in zip(fig.get_axes(), RULES):
        done = lines_labelled(ax, "does")
        assert len(done) == 1
        np.testing.assert_allclose(done[0].get_xdata(), [row["batch"] for row in rows])
        np.testing.assert_allclose(done[0].get_ydata(), [row["done"][rule] for row in rows])
        for k in range(2):
            lost = lines_labelled(ax, f"round {k + 1} hears only itself")
            assert len(lost) == 1, f"{rule}, round {k + 1}"
            np.testing.assert_allclose(lost[0].get_ydata(), [row["blocks"][k]["lost"][rule] for row in rows])


def test_readability_is_drawn_when_given_in_the_colour_of_its_round():
    fig = plot_ablation.plot_ablation(fake_curve(), readability=fake_readability())
    read = fake_readability()["curve"]
    rows = [row for row in read if row["batch"] > 0]
    for ax, rule in zip(fig.get_axes(), RULES):
        for k in range(2):
            readable = lines_labelled(ax, f"readable in draft {k + 1}")
            lost = lines_labelled(ax, f"round {k + 1} hears only itself")
            assert len(readable) == 1
            np.testing.assert_allclose(readable[0].get_ydata(), [row["scores"][rule][k + 1] for row in rows])
            assert readable[0].get_color() == lost[0].get_color()
            assert readable[0].get_linestyle() == "--"


def test_no_readability_lines_without_the_file():
    fig = plot_ablation.plot_ablation(fake_curve())
    for ax in fig.get_axes():
        assert lines_labelled(ax, "readable") == []


def test_log_by_default_and_linear_puts_batch_0_on_the_axis():
    assert all(ax.get_xscale() == "log" for ax in plot_ablation.plot_ablation(fake_curve()).get_axes())
    fig = plot_ablation.plot_ablation(fake_curve(), linear=True)
    for ax in fig.get_axes():
        assert ax.get_xscale() == "linear"
        assert lines_labelled(ax, "does")[0].get_xdata()[0] == 0


def test_the_end_of_pass_1_is_marked():
    fig = plot_ablation.plot_ablation(fake_curve())
    last = fake_curve()["curve"][-1]
    for ax in fig.get_axes():
        dotted = [line for line in ax.get_lines() if line.get_linestyle() == ":"]
        assert len(dotted) == 1
        assert dotted[0].get_xdata()[0] == pytest.approx(last["batch"] / last["pass"])


def test_main_writes_a_picture(tmp_path):
    curve_file = tmp_path / "curve.json"
    curve_file.write_text(json.dumps(fake_curve()))
    read_file = tmp_path / "read.json"
    read_file.write_text(json.dumps(fake_readability()))
    out = tmp_path / "ablation.png"
    plot_ablation.main([str(curve_file), "--readability", str(read_file), "--out", str(out)])
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
