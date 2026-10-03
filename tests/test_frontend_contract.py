import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"


def test_frontend_is_complete_and_wired_without_external_assets() -> None:
    html = (FRONTEND / "index.html").read_text(encoding="utf-8")
    css = (FRONTEND / "styles.css").read_text(encoding="utf-8")
    javascript = (FRONTEND / "app.js").read_text(encoding="utf-8")

    assert '<link rel="stylesheet" href="./styles.css"' in html
    assert '<script type="module" src="./app.js"></script>' in html
    assert 'data-view="plan"' in html
    assert 'data-view="capabilities"' in html
    assert 'data-view="evidence"' in html
    assert 'data-view="boundaries"' in html
    assert "prefers-reduced-motion" in css
    assert "canopy-foundry-execution-plan.v1" in javascript

    # The public frontend is a file reader, not a second Foundry or Hub API.
    for forbidden in ("fetch(", "XMLHttpRequest", "WebSocket", "EventSource", "http://", "https://"):
        assert forbidden not in javascript
    assert ("IDA" + "-TRAIN-V2") not in html + css + javascript
    assert "No network calls" in html
    assert "does not independently verify" in (FRONTEND / "README.md").read_text(encoding="utf-8")

    sprite = json.loads((FRONTEND / "assets" / "ida-cli-sprite.json").read_text(encoding="utf-8"))
    assert sprite["WALK"]["mode"] == "quadrant"
    assert len(sprite["WALK"]["frames"]) == 4
    assert sprite["IDA"]["frames"]


def test_frontend_does_not_present_the_example_as_runtime_evidence() -> None:
    html = (FRONTEND / "index.html").read_text(encoding="utf-8")
    javascript = (FRONTEND / "app.js").read_text(encoding="utf-8")

    assert "EXAMPLE ONLY" in javascript
    assert "non-runtime data" in html
    assert "not a runtime receipt" in javascript
    assert "runtime_verification_required" in javascript
    assert "observed" in javascript
    assert "persisted" in javascript
