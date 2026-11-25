import http.client
import json
import math
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from panomotion import sphere
from panomotion.server import Handler


def _write_motion_pair(run_dir, i=0, width=240, height=120,
                       cols=12, rows=6, omega=None):
    if omega is None:
        omega = np.array([0.004, -0.002, 0.003])
    grid = np.broadcast_to(omega, (rows, cols, 3)).copy()
    doc = {
        "width": width,
        "height": height,
        "omega_grid": np.round(grid, 8).tolist(),
        "global_omega": np.round(omega, 8).tolist(),
    }
    (run_dir / "data").mkdir(parents=True, exist_ok=True)
    (run_dir / "data" / f"pair_{i:03d}.json").write_text(
        json.dumps(doc), encoding="utf-8")


@pytest.fixture()
def server(tmp_path):
    (tmp_path / "demo").mkdir()
    (tmp_path / "demo" / "meta.json").write_text("{}", encoding="utf-8")

    class H(Handler):
        runs_dir = tmp_path

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv.server_address[1], tmp_path
    srv.shutdown()
    srv.server_close()


def get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as r:
        return r.status, r.read()


def get_any(port, path):
    try:
        return get(port, path)
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_frontend_and_runs_api(server):
    port, _ = server
    status, body = get(port, "/")
    assert status == 200 and "app.js" in body.decode("utf-8")
    status, body = get(port, "/app.js")
    assert status == 200 and b"renderSphere" in body
    status, body = get(port, "/sphere_transform.js")
    assert status == 200 and b"rodrigues" in body
    status, body = get(port, "/api/runs")
    runs = json.loads(body)["runs"]
    assert any(r["name"] == "demo" for r in runs)


def test_traversal_is_forbidden(server):
    port, _ = server
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", "/artifacts/demo/../../etc/passwd")
    resp = conn.getresponse()
    resp.read()
    assert resp.status in (403, 404)
    conn.close()


def test_motion_api_advances_real_omega_field(server):
    port, tmp = server
    omega = np.array([0.004, -0.002, 0.003])
    _write_motion_pair(tmp / "demo", omega=omega)
    d = sphere.normalize(np.array([0.6, -0.7, 0.2]))
    q = ",".join(f"{x:.8f}" for x in d)
    status, body = get(port, f"/api/motion?name=demo&i=0&dir={q}")
    assert status == 200
    doc = json.loads(body)
    moved = np.asarray(doc["moved_dir"][0])
    expected = sphere.rodrigues(omega, d[None, :])[0]
    # The response comes from the stored grid + pixel interpolation, not fixed
    # data: it matches the Rodrigues transport of the actual omega field.
    assert np.rad2deg(sphere.geodesic_angle(
        moved[None], expected[None]))[0] < 1e-4
    expected_arc = np.rad2deg(sphere.geodesic_angle(d[None], moved[None]))[0]
    assert doc["arc_deg"][0] == pytest.approx(float(expected_arc), abs=1e-4)
    assert doc["arc_deg"][0] > 0.0
    assert np.allclose(np.rad2deg(omega), doc["global_omega_deg"], atol=1e-5)


def test_motion_api_seam_direction_keeps_small_arc(server):
    port, tmp = server
    # Motion crossing the +/-pi seam: longitude wraps, geodesic arc is small.
    d0 = sphere.lonlat_to_direction(math.pi - math.radians(3), 0.2)
    d1 = sphere.lonlat_to_direction(-math.pi + math.radians(3), 0.2)
    omega = sphere.rotation_vector_between(d0[None, :], d1[None, :])[0]
    _write_motion_pair(tmp / "demo", omega=omega)
    q = ",".join(f"{x:.8f}" for x in d0)
    status, body = get(port, f"/api/motion?name=demo&i=0&dir={q}")
    doc = json.loads(body)
    assert 0.0 < doc["arc_deg"][0] < 7.0


def test_motion_api_hotspot_and_error_paths(server):
    port, tmp = server
    _write_motion_pair(tmp / "demo")
    d = sphere.normalize(np.array([0.1, 0.2, 0.97]))
    (tmp / "demo" / "hotspots.json").write_text(json.dumps({
        "hotspots": [{
            "hotspot_id": 3,
            "frames": {"0": {"dir": np.round(d, 8).tolist()}},
        }]
    }), encoding="utf-8")
    status, body = get(port, "/api/motion?name=demo&i=0&hotspot=3")
    assert status == 200
    doc = json.loads(body)
    assert len(doc["moved_dir"]) == 1 and doc["arc_deg"][0] > 0.0

    status, body = get_any(port, "/api/motion?name=demo&i=0&hotspot=99")
    assert status == 404
    status, body = get_any(port, "/api/motion?name=demo&i=0")
    assert status == 400
    status, body = get(port, "/api/hotspots?name=demo")
    assert status == 200 and json.loads(body)["hotspots"][0]["hotspot_id"] == 3
    status, body = get_any(port, "/api/hotspots?name=missing")
    assert status == 404
