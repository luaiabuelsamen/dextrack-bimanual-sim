"""Action chunking and temporal ensembling, without needing torch."""
import numpy as np
import pytest

from oppdef.policy import (make_chunks, chunk_dataset, TemporalEnsembler, Norm,
                           Scripted)


def test_chunks_are_the_next_h_actions():
    o = np.arange(6, dtype=float)[:, None]
    a = np.arange(6, dtype=float)[:, None] * 10
    _x, y = make_chunks(o, a, 3)
    assert y.shape == (6, 3, 1)
    assert y[0, :, 0].tolist() == [0, 10, 20]
    assert y[1, :, 0].tolist() == [10, 20, 30]


def test_chunk_tail_holds_the_last_action():
    """The expert holds its final pose, so repeating it is the true
    continuation -- not padding with a filler value."""
    a = np.arange(4, dtype=float)[:, None] * 10
    _x, y = make_chunks(a, a, 3)
    assert y[-1, :, 0].tolist() == [30, 30, 30]
    assert y[-2, :, 0].tolist() == [20, 30, 30]


def test_episodes_are_chunked_separately():
    """A window that runs off one episode into the next teaches the policy to
    follow a demonstration with a different peg pose -- a corruption that looks
    like ordinary training noise."""
    e1 = (np.zeros((4, 2)), np.full((4, 1), 1.0))
    e2 = (np.zeros((4, 2)), np.full((4, 1), 9.0))
    _X, Y = chunk_dataset([e1, e2], 3)
    assert len(Y) == 8
    assert np.all(Y[:4] == 1.0), "episode 1 chunk leaked into episode 2"
    assert np.all(Y[4:] == 9.0)


def test_make_chunks_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        make_chunks(np.zeros((5, 2)), np.zeros((4, 1)), 2)


# -- the ensembler ----------------------------------------------------------
def test_ensembler_uses_element_i_of_the_chunk_from_i_steps_ago():
    """The prediction about NOW made i steps ago is element i of that chunk.

    Keeping only each chunk's first element makes this a moving average of
    fresh predictions instead -- it ran that way at first, and with identical
    chunks [1,2,3,4] it settled on 1.0 where the answer is 2.5.
    """
    c = np.array([[1.0], [2.0], [3.0], [4.0]])
    e = TemporalEnsembler(4, 1, m=0.0)
    out = [float(e.step(c)[0]) for _ in range(6)]
    assert out[0] == pytest.approx(1.0)          # only the fresh one is live
    assert out[3] == pytest.approx(2.5)          # (1+2+3+4)/4 once filled
    assert out[-1] == pytest.approx(2.5)


def test_large_m_uses_the_newest_chunk_only():
    c = np.array([[1.0], [2.0], [3.0], [4.0]])
    e = TemporalEnsembler(4, 1, m=50.0)
    assert float(e.step(c)[0]) == pytest.approx(1.0)
    for _ in range(5):
        v = float(e.step(c)[0])
    assert v == pytest.approx(1.0, abs=1e-6)


def test_horizon_one_is_a_passthrough():
    e = TemporalEnsembler(1, 2, m=0.0)
    for _ in range(3):
        assert np.allclose(e.step(np.array([[7.0, -3.0]])), [7.0, -3.0])


def test_reset_clears_history():
    c = np.array([[1.0], [2.0]])
    e = TemporalEnsembler(2, 1, m=0.0)
    e.step(c); e.step(c)
    e.reset()
    assert float(e.step(c)[0]) == pytest.approx(1.0)


# -- normalisation ----------------------------------------------------------
def test_norm_roundtrips():
    rng = np.random.default_rng(0)
    O, A = rng.normal(size=(50, 4)), rng.normal(size=(50, 3)) * 5 + 2
    n = Norm.fit(O, A)
    assert np.allclose(n.y_inv((A - n.am) / n.as_), A, atol=1e-6)
    assert np.allclose(n.x(O).mean(0), 0, atol=1e-6)


def test_scripted_wraps_a_callable():
    p = Scripted(lambda o: np.array([1.0, 2.0]), act_dim=2)
    p.reset()
    assert p(np.zeros(3)).tolist() == [1.0, 2.0]
    assert p.act_dim() == 2
