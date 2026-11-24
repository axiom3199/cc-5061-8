import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from panomotion.server import Handler


@pytest.fixture()
def server(tmp_path):
    (tmp_path / "demo").mkdir()
    (tmp_path / "demo" / "meta.json").write_text("{}", encoding="utf-8")

    class H(Handler):
        runs_dir = tmp_path

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv.server_address[1]
    srv.shutdown()
    srv.server_close()


def get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as r:
        return r.status, r.read()


def test_frontend_and_runs_api(server):
    status, body = get(server, "/")
    assert status == 200 and "app.js" in body.decode("utf-8")
    status, body = get(server, "/app.js")
    assert status == 200 and b"renderSphere" in body
    status, body = get(server, "/api/runs")
    runs = json.loads(body)["runs"]
    assert any(r["name"] == "demo" for r in runs)


def test_traversal_is_forbidden(server):
    import http.client
    conn = http.client.HTTPConnection("127.0.0.1", server, timeout=5)
    conn.request("GET", "/artifacts/demo/../../etc/passwd")
    resp = conn.getresponse()
    resp.read()
    assert resp.status in (403, 404)
    conn.close()
