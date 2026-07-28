from regression_260707.monitoring import runtime_transition
from regression_260707.monitoring.release_canary import _sha256_file
from regression_260707.monitoring.runtime_transition import (
    _require_process_location,
    _wait_for_runtime_nsga_generation,
)


def test_windows_venv_base_interpreter_is_valid_rollback_process(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    venv = tmp_path / "venv"
    scripts = venv / "Scripts"
    scripts.mkdir(parents=True)
    launcher = scripts / "python.exe"
    launcher.write_bytes(b"venv-launcher")
    base = tmp_path / "base" / "python.exe"
    base.parent.mkdir()
    base.write_bytes(b"base-interpreter")
    (venv / "pyvenv.cfg").write_text(
        f"home = {base.parent}\nexecutable = {base}\n",
        encoding="utf-8",
    )
    process = {
        "executable": str(base),
        "executable_sha256": _sha256_file(base, 1024),
        "cwd": str(source),
        "command": [str(base), "-m", "uvicorn"],
    }

    _require_process_location(process, source, venv)


def test_runtime_nsga_gate_receives_authenticated_current7(monkeypatch):
    current7 = {"path": "current7-index.json", "sha256": "a" * 64}
    expected = ({}, {}, {}, {}, {})
    observed = {}

    def fake_wait(base_url, **kwargs):
        observed["base_url"] = base_url
        observed.update(kwargs)
        return expected

    monkeypatch.setattr(
        runtime_transition,
        "_wait_for_stable_nsga_generation",
        fake_wait,
    )

    result = _wait_for_runtime_nsga_generation(
        "http://127.0.0.1:8010",
        {
            "current7_index": current7,
            "blocker_hpo_v2_status": None,
        },
    )

    assert result is expected
    assert observed == {
        "base_url": "http://127.0.0.1:8010",
        "expected_current7_index": current7,
        "expected_current7_condition_indexes": None,
    }


def test_runtime_nsga_gate_receives_current7_condition_indexes(monkeypatch):
    conditions = [
        {"path": "entry-index.json", "sha256": "a" * 64},
        {"path": "final-index.json", "sha256": "b" * 64},
    ]
    expected = ({}, {}, {}, {}, {})
    observed = {}

    def fake_wait(base_url, **kwargs):
        observed["base_url"] = base_url
        observed.update(kwargs)
        return expected

    monkeypatch.setattr(
        runtime_transition,
        "_wait_for_stable_nsga_generation",
        fake_wait,
    )
    result = _wait_for_runtime_nsga_generation(
        "http://127.0.0.1:8010",
        {
            "current7_index": None,
            "current7_condition_indexes": conditions,
            "blocker_hpo_v2_status": None,
        },
    )

    assert result is expected
    assert observed == {
        "base_url": "http://127.0.0.1:8010",
        "expected_current7_index": None,
        "expected_current7_condition_indexes": conditions,
    }
