"""The penetration probe, on synthetic geometry with known answers.

A hand of one box link and one sphere link, an object that is one cube of
half-width 0.05 m. Poses are chosen so the true maximum depth is known:
a box corner pushed d metres into a cube face has depth d; a sphere whose
centre is r - d from a face has depth d. The tests pin the three
equivalences the training signal rests on: dense grid vs the exact
answer, the signed-depth volume vs the plane test, and chunked vs
unchunked batches. They need torch and trimesh, no assets.
"""
from __future__ import annotations

import numpy as np
import pytest
import trimesh

torch = pytest.importorskip("torch")
from dextrack import penetration_torch as pt  # noqa: E402

URDF = """<?xml version="1.0"?>
<robot name="two_links">
  <link name="box_link">
    <collision><origin xyz="0 0 0" rpy="0 0 0"/><geometry><box size="0.02 0.02 0.04"/></geometry></collision>
  </link>
  <link name="ball_link">
    <collision><origin xyz="0 0 0" rpy="0 0 0"/><geometry><sphere radius="0.01"/></geometry></collision>
  </link>
  <link name="bare_link"/>
</robot>
"""
HALF = 0.05
NAMES = ["box_link", "ball_link", "bare_link"]
IDENT = [0.0, 0.0, 0.0, 1.0]           # xyzw


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    d = tmp_path_factory.mktemp("probe")
    (d / "hand.urdf").write_text(URDF)
    obj = d / "obj"
    obj.mkdir()
    trimesh.creation.box(extents=[2 * HALF] * 3).export(str(obj / "decomposed.obj"))
    return d / "hand.urdf", obj


def states(box_pos, ball_pos):
    """(1, 3, 13) rigid-body states: pos, quat xyzw, zeros."""
    rb = torch.zeros(1, 3, 13)
    rb[0, 0, :3] = torch.tensor(box_pos); rb[0, 0, 3:7] = torch.tensor(IDENT)
    rb[0, 1, :3] = torch.tensor(ball_pos); rb[0, 1, 3:7] = torch.tensor(IDENT)
    rb[0, 2, 3:7] = torch.tensor(IDENT)
    return rb


OBJ = torch.tensor([[0.0, 0.0, 0.0] + IDENT])


def test_dense_grid_recovers_known_depths(scene):
    urdf, obj = scene
    probe = pt.PenetrationProbe(str(urdf), str(obj), NAMES, "cpu", spacing=0.002)
    # box (half-height 0.02) with its bottom face 3 mm below the cube's top face; ball far away
    rb = states([0.0, 0.0, HALF + 0.02 - 0.003], [0.0, 0.0, 1.0])
    depth, n_touch, per_link = probe(rb, OBJ)
    assert abs(depth.item() - 0.003) < 1e-4
    assert n_touch.item() == 1
    # ball (r = 0.01) with its centre 8 mm above the top face: 2 mm inside
    rb = states([0.0, 0.0, 1.0], [0.0, 0.0, HALF + 0.008])
    depth, _, _ = probe(rb, OBJ)
    assert abs(depth.item() - 0.002) < 3e-4          # icosphere vertices, not a perfect sphere
    # nothing inside
    rb = states([0.0, 0.0, 1.0], [0.0, 0.0, 1.0])
    depth, n_touch, _ = probe(rb, OBJ)
    assert depth.item() == 0.0 and n_touch.item() == 0


def test_sparse_probe_misses_an_edge_depth_the_dense_grid_finds(scene):
    """A box edge across the cube's edge: the deepest point is mid-edge, away
    from every corner. This is the geometry a policy exploited."""
    urdf, obj = scene
    sparse = pt.PenetrationProbe(str(urdf), str(obj), NAMES, "cpu", spacing=0.0)
    dense = pt.PenetrationProbe(str(urdf), str(obj), NAMES, "cpu", spacing=0.002)
    # box rotated 45 deg about x, centred over the cube's +y,+z edge, sunk 4 mm
    s = np.sin(np.pi / 8); c = np.cos(np.pi / 8)
    rb = torch.zeros(1, 3, 13)
    rb[0, 0, :3] = torch.tensor([0.0, HALF, HALF - 0.004]); rb[0, 0, 3:7] = torch.tensor([s, 0.0, 0.0, c])
    rb[0, 1, :3] = torch.tensor([0.0, 0.0, 1.0]); rb[0, 1, 3:7] = torch.tensor(IDENT); rb[0, 2, 3:7] = torch.tensor(IDENT)
    d_sparse = sparse(rb, OBJ)[0].item(); d_dense = dense(rb, OBJ)[0].item()
    assert d_dense > 0.003
    assert d_dense >= d_sparse           # the dense grid never reads less than the sparse set


def test_volume_matches_plane_test(scene):
    urdf, obj = scene
    planes = pt.PenetrationProbe(str(urdf), str(obj), NAMES, "cpu", spacing=0.002)
    volume = pt.PenetrationProbe(str(urdf), str(obj), NAMES, "cpu", spacing=0.002, sdf_res=0.001)
    rng = np.random.default_rng(0)
    for _ in range(20):
        box = [rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03), HALF + rng.uniform(0.005, 0.03)]
        ball = [rng.uniform(-0.03, 0.03), HALF + rng.uniform(0.0, 0.012), rng.uniform(-0.03, 0.03)]
        rb = states(box, ball)
        a = planes(rb, OBJ); b = volume(rb, OBJ)
        assert abs(a[0].item() - b[0].item()) < 3e-4               # within a third of a voxel
        assert torch.equal(a[1], b[1]) or abs(a[0].item()) < 1e-3   # touch counts agree except at the surface


def test_chunking_is_invisible(scene):
    urdf, obj = scene
    probe = pt.PenetrationProbe(str(urdf), str(obj), NAMES, "cpu", spacing=0.003, sdf_res=0.001)
    rng = np.random.default_rng(1)
    rb = torch.cat([states([rng.uniform(-0.02, 0.02), 0.0, HALF + rng.uniform(0.0, 0.03)],
                           [0.0, rng.uniform(-0.02, 0.02), HALF + rng.uniform(0.0, 0.012)]) for _ in range(37)])
    obj_pose = OBJ.repeat(37, 1)
    whole = probe(rb, obj_pose, chunk=1000)
    parts = probe(rb, obj_pose, chunk=8)
    for w, p in zip(whole, parts):
        assert torch.allclose(w, p)
