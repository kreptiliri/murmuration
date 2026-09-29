import json

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
plot_readability = pytest.importorskip("plot_readability")

RULES = ("separation", "alignment", "cohesion")
N_DRAFTS = 3


def fake_curve():
    #made-up scores that differ by rule, draft and batch, so a mix-up shows
    batches = [0, 1, 2, 4, 8, 100, 1000, 40950]
    curve = []
    for i, batch in enumerate(batches):
        scores = {}
        for r, rule in enumerate(RULES):
            scores[rule] = [round(0.1 * r + 0.01 * k + 0.05 * i, 4) for k in range(N_DRAFTS)]
        curve.append({"batch": batch, "pass": batch / 1365, "scores": scores})
    return {"curve": curve, "learn_clips": 50, "check_clips": 20, "alpha": 1.0}


def solid_lines(ax):
    return [line for line in ax.get_lines() if line.get_linestyle() == "-"]


def dashed_lines(ax):
    return [line for line in ax.get_lines() if line.get_linestyle() == "--"]


# --- the layout ------------------------------------------------------------------------

def test_one_panel_per_rule_named_after_it():
    fig = plot_readability.plot_readability(fake_curve())
    axes = fig.get_axes()
    assert len(axes) == 3
    for ax, rule in zip(axes, RULES):
        assert rule in (ax.get_title() + ax.get_title(loc="left")).lower()


def test_one_curve_and_one_dashed_untrained_line_per_draft_in_each_panel():
    fig = plot_readability.plot_readability(fake_curve())
    for ax in fig.get_axes():
        assert len(solid_lines(ax)) == N_DRAFTS
        assert len(dashed_lines(ax)) == N_DRAFTS


# --- the lines are the right numbers --------------------------------------------------------

def test_each_curve_is_that_rule_and_draft_over_training():
    """Log scale by default, so batch 0 cannot sit on the axis: the curve starts
    at batch 1 and batch 0 is the dashed line."""
    data = fake_curve()
    fig = plot_readability.plot_readability(data)
    for ax, rule in zip(fig.get_axes(), RULES):
        for k, line in enumerate(solid_lines(ax)):
            rows = [row for row in data["curve"] if row["batch"] > 0]
            np.testing.assert_allclose(line.get_xdata(), [row["batch"] for row in rows])
            np.testing.assert_allclose(line.get_ydata(), [row["scores"][rule][k] for row in rows],
                                       err_msg=f"{rule}, draft {k}")


def test_the_dashed_line_is_the_untrained_score():
    """The free part: what a random network already shows. The gap between the
    curve and this line is what training added."""
    data = fake_curve()
    untrained = data["curve"][0]
    assert untrained["batch"] == 0
    fig = plot_readability.plot_readability(data)
    for ax, rule in zip(fig.get_axes(), RULES):
        for k, line in enumerate(dashed_lines(ax)):
            assert set(np.round(line.get_ydata(), 6)) == {round(untrained["scores"][rule][k], 6)}


def test_a_curve_and_its_dashed_line_share_a_colour():
    fig = plot_readability.plot_readability(fake_curve())
    for ax in fig.get_axes():
        for solid, dashed in zip(solid_lines(ax), dashed_lines(ax)):
            assert solid.get_color() == dashed.get_color()


# --- the x-axis --------------------------------------------------------------------------------

def test_the_x_axis_is_log_by_default():
    fig = plot_readability.plot_readability(fake_curve())
    assert all(ax.get_xscale() == "log" for ax in fig.get_axes())


def test_linear_puts_batch_0_on_the_axis_too():
    data = fake_curve()
    fig = plot_readability.plot_readability(data, linear=True)
    for ax in fig.get_axes():
        assert ax.get_xscale() == "linear"
        for line in solid_lines(ax):
            assert line.get_xdata()[0] == 0
            assert len(line.get_xdata()) == len(data["curve"])


# --- main ------------------------------------------------------------------------------------------

def test_main_reads_the_sweep_file_and_writes_a_picture(tmp_path):
    curve_file = tmp_path / "curve.json"
    curve_file.write_text(json.dumps(fake_curve()))
    out = tmp_path / "readability.png"
    plot_readability.main([str(curve_file), "--out", str(out)])
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_main_takes_linear(tmp_path):
    curve_file = tmp_path / "curve.json"
    curve_file.write_text(json.dumps(fake_curve()))
    out = tmp_path / "readability.png"
    fig = plot_readability.main([str(curve_file), "--out", str(out), "--linear"])
    assert all(ax.get_xscale() == "linear" for ax in fig.get_axes())


# --- reading the x-axis without doing sums ----------------------------------------------

def dotted_lines(ax):
    return [line for line in ax.get_lines() if line.get_linestyle() == ":"]


def test_the_end_of_pass_1_is_marked_in_every_panel():
    """Batches alone do not say how far into training a point is. 1365 batches
    is one pass over the data at batch 256; the line is put where the curve
    itself says a pass ends (batch / pass), not at a number typed in here."""
    data = fake_curve()
    per_pass = data["curve"][-1]["batch"] / data["curve"][-1]["pass"]
    fig = plot_readability.plot_readability(data)
    for ax in fig.get_axes():
        marks = dotted_lines(ax)
        assert len(marks) == 1
        assert set(np.round(marks[0].get_xdata(), 6)) == {round(per_pass, 6)}


def test_the_log_axis_says_1000_not_ten_to_the_three():
    fig = plot_readability.plot_readability(fake_curve())
    fig.canvas.draw()
    labels = [t.get_text() for t in fig.get_axes()[0].get_xticklabels() if t.get_text()]
    assert "1,000" in labels, labels
    assert not any("^" in label for label in labels), labels
