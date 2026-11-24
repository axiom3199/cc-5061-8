import json

from panomotion.pipeline import PipelineConfig, run_pipeline
from panomotion.synth import generate_synthetic


def test_pipeline_smoke_outputs_contract(tmp_path):
    seq = generate_synthetic(width=480, height=240, n_frames=4,
                            seed=3, verbose=False)
    pcfg = PipelineConfig(name="smoke", cols=20, rows=10, n_frames=4,
                          width=480, height=240, run_baseline=True, fps=4)
    run_dir = tmp_path / "run"
    run_pipeline(seq.frames, run_dir, seq=seq, pcfg=pcfg,
                 progress=lambda *a: None)

    report = json.loads((run_dir / "report.json").read_text())
    assert report["n_frames"] == 4
    assert len(report["pairs"]) == 3
    assert report["has_ground_truth"] is True
    for key in ("mean_angle_error_median_deg", "mean_reprojection_psnr",
                "planar_baseline_mean", "total_anomalies"):
        assert key in report["summary"]

    tracks = json.loads((run_dir / "tracks.json").read_text())
    assert isinstance(tracks, list) and len(tracks) >= 20
    polar = next(t for t in tracks if t["name"] == "polar-crosser")
    assert polar["crosses_pole"] is True
    assert polar["median_error_deg"] < 1.0
    assert polar["median_error_deg"] < 1.0

    pair = json.loads((run_dir / "data" / "pair_000.json").read_text())
    for key in ("target_lon", "target_lat", "cls", "seam", "seam_agg",
                "anomalies", "reprojection", "gt_lon", "error_deg"):
        assert key in pair
    n = 20 * 10
    assert len(pair["target_lon"]) == n

    for fname in ("original.mp4", "estimated_spherical_motion.mp4",
                  "ground_truth_motion.mp4", "reprojection_check.mp4",
                  "reprojection_check.gif"):
        assert (run_dir / "videos" / fname).exists()
        assert (run_dir / "videos" / fname).stat().st_size > 1000
