from pathlib import Path

from scripts.env.deploy_vllm_overlay import RUNTIME_FILES, atomic_copy


def test_runtime_overlay_file_set_excludes_tests() -> None:
    assert len(RUNTIME_FILES) == 7
    assert all(path.startswith("vllm/") for path in RUNTIME_FILES)
    assert not any(path.startswith("tests/") for path in RUNTIME_FILES)


def test_atomic_copy_replaces_target(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    target = tmp_path / "package" / "target.py"
    source.write_text("new\n", encoding="utf-8")
    target.parent.mkdir()
    target.write_text("old\n", encoding="utf-8")

    atomic_copy(source, target)

    assert target.read_text(encoding="utf-8") == "new\n"
