"""A GRAB sequence name does not identify a clip. Seeds must carry the subject.

Eighty sequence names in the inventory exist under more than one subject, and
the two recordings are different data: `camera_takepicture_2` holds for 161
frames under s1 and 76 under s2. A dict keyed on the bare name silently keeps
one of them, so a stage that looks its seed up by name can be handed a grasp
belonging to a different recording. That is not hypothetical -- seven of ten
references in the first seeded stage-3 run trained with another subject's wrist
offset, and it stayed invisible for two hours because every number it produced
was plausible.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

INV = Path("results/grab_inventory.json")


def _rows():
    if not INV.exists():
        pytest.skip("inventory not built")
    return json.loads(INV.read_text())["rows"]


def test_sequence_names_really_do_collide():
    """The hazard is real, so the guards below are not decoration."""
    rows = _rows()
    seen = {}
    for r in rows:
        seen.setdefault(r["seq"], set()).add(r["subject"])
    collide = {k: v for k, v in seen.items() if len(v) > 1}
    assert len(collide) > 1, (
        "expected many-subject sequence names in GRAB; if this fails the "
        "inventory changed and the subject-keying guards need rechecking")


def test_colliding_names_are_different_clips():
    """Not merely duplicated rows: the two recordings differ in length."""
    rows = _rows()
    by = {}
    for r in rows:
        by.setdefault(r["seq"], []).append(r)
    differing = 0
    for name, group in by.items():
        if len(group) < 2:
            continue
        lens = {r.get("rhand_hold_len") for r in group}
        if len(lens) > 1:
            differing += 1
    assert differing > 0, "colliding names should describe different clips"


@pytest.mark.parametrize("path", [
    "results/stage2_grips.json",
    "results/stage2_grips_g9.json",
    "results/stage2_grips_clean.json",
])
def test_seed_files_carry_a_subject(path):
    """Every seed row must say which recording it belongs to."""
    p = Path(path)
    if not p.exists():
        pytest.skip(f"{path} not produced yet")
    rows = json.loads(p.read_text())["rows"]
    assert rows, f"{path} is empty"
    for r in rows:
        assert r.get("subject"), f"{path}: a seed row without a subject: {r.get('seq')}"


def test_seed_lookups_are_subject_qualified():
    """No experiment may key a seed dict on the bare sequence name."""
    bad = []
    for p in sorted(Path("experiments").rglob("*.py")):
        src = p.read_text()
        for pat in ('{x["seq"]: x', "{x['seq']: x",
                    '{r["seq"]: r', "{r['seq']: r"):
            if pat in src:
                bad.append(f"{p}: {pat}")
    assert not bad, (
        "seed/inventory dicts keyed on the bare sequence name; use "
        "f\"{row['subject']}/{row['seq']}\" -- see this module's docstring:\n  "
        + "\n  ".join(bad))
