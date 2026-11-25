"""Hotspot editor and motion-driven stability tests.

A hotspot is one spherical object.  Both projection outlets may split its
outline, but the editor must recognise every fragment via the hotspot id and
picking on either side of the seam (or after a pole crossing) returns the same
hotspot.  The per-frame chain is driven by actual spherical matching, so its
ground-truth angular error is measured rather than asserted.
"""

import numpy as np
import pytest

from panomotion import sphere
from panomotion.hotspots import Hotspot, HotspotEditor
from panomotion.projection import (EquirectProjection, PerspectiveProjection,
                                   fragment_object_ids)
from panomotion.synth import generate_synthetic
from panomotion.tracking import track_hotspot_anchor


def test_editor_keeps_one_identity_for_seam_split_outline():
    editor = HotspotEditor(motions=[])
    center = sphere.lonlat_to_direction(np.pi - 0.01, 0.0)
    hs = editor.add(center, name="seam", radius_deg=5.0)
    eq = EquirectProjection(480, 240)
    frags = editor.render(eq, frame=0)
    assert len(frags) >= 2  # split across the plate borders
    assert fragment_object_ids(frags) == {hs.hotspot_id}
    assert len(editor.hotspots) == 1  # no duplicate hotspot at the seam


def test_pick_on_both_seam_sides_resolves_same_hotspot():
    editor = HotspotEditor(motions=[])
    center = sphere.lonlat_to_direction(np.pi - 0.02, 0.1)
    hs = editor.add(center, radius_deg=6.0)
    # Click directions a couple of degrees either side of the +/-pi seam.
    click_a = sphere.lonlat_to_direction(np.pi - 0.04, 0.1)
    click_b = sphere.lonlat_to_direction(-np.pi + 0.04, 0.1)
    assert editor.pick(click_a) is hs
    assert editor.pick(click_b) is hs


def test_clip_hotspot_equivalent_across_two_projections():
    editor = HotspotEditor(motions=[])
    center = sphere.lonlat_to_direction(2.4, 0.6)
    hs = editor.add(center, radius_deg=5.0)
    eq = EquirectProjection(480, 240)
    persp = PerspectiveProjection.look_at(240, 120, center, np.deg2rad(120.0))
    verdict = editor.clip_hotspot(
        hs.hotspot_id, {"equirect": eq, "perspective": persp})
    assert verdict["equivalent"] is True
    assert verdict["object_ids"] == [hs.hotspot_id]


def test_hotspot_outline_is_a_geodesic_circle_of_constant_radius():
    center = sphere.lonlat_to_direction(-1.0, 0.4)
    hs = Hotspot(1, "h", center, np.deg2rad(7.0))
    outline = hs.outline_dirs(0, samples=80)
    ang = np.rad2deg(sphere.geodesic_angle(
        np.broadcast_to(center, outline.shape), outline))
    assert np.allclose(ang, 7.0, atol=0.05)


@pytest.fixture(scope="module")
def sequence():
    return generate_synthetic(width=480, height=240, n_frames=6,
                              seed=11, verbose=False)


def test_motion_driven_object_hotspot_tracks_ground_truth(sequence):
    seq = sequence
    global_Rs = [seq.background_matrix(t) for t in range(seq.n_frames - 1)]
    for obj in seq.objects:
        anchor = sphere.normalize(seq.Rs[0].T @ obj.centers[0])
        chain, weak = track_hotspot_anchor(
            seq.frames, anchor, global_Rs)
        gt_chain = np.stack([
            sphere.normalize(seq.Rs[t].T @ obj.centers[t])
            for t in range(seq.n_frames)])
        err = np.rad2deg(sphere.geodesic_angle(chain, gt_chain))
        # Real measured accuracy on the two stress objects, including the
        # pole crosser and the seam crosser.
        assert np.median(err) < 1.0
        assert err.max() < 2.0


def test_pipeline_hotspot_document_carries_evidence(sequence, tmp_path):
    from panomotion.pipeline import PipelineConfig, run_pipeline
    import json
    pcfg = PipelineConfig(name="h", cols=20, rows=10, n_frames=6,
                          width=480, height=240, run_baseline=False, fps=4)
    run_pipeline(sequence.frames, tmp_path / "run", seq=sequence,
                 pcfg=pcfg, progress=lambda *a: None)
    doc = json.loads((tmp_path / "run" / "hotspots.json").read_text())
    assert len(doc["hotspots"]) == 2
    for check in doc["projection_checks"]:
        assert check["same_spherical_object"] is True
        assert 0.0 < check["valid_region_coverage"] <= 1.0
        # One id per outlet even when the equirect outlet splits fragments.
        assert len(check["object_ids_equirect"]) == 1
        assert check["object_ids_equirect"] == check["object_ids_perspective"]
    for gt in doc["gt_angular_error"]:
        assert gt["median_error_deg"] < 1.0
        assert gt["valid_fraction"] == 1.0
    # Boundary velocity evidence exists for the polar-crossing hotspot.
    polar = next(h for h in doc["hotspots"] if h["name"] == "polar-crosser")
    flags = [b["crosses_pole"] for b in polar["trajectory"]["boundary"]]
    assert any(flags)
