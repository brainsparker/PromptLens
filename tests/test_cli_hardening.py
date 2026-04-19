from pathlib import Path

from promptlens.cli import _remove_path_if_exists


def test_remove_path_if_exists_removes_file(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_text("x")

    _remove_path_if_exists(target)

    assert not target.exists()


def test_remove_path_if_exists_removes_directory(tmp_path: Path) -> None:
    target_dir = tmp_path / "latest"
    target_dir.mkdir()
    (target_dir / "nested.txt").write_text("x")

    _remove_path_if_exists(target_dir)

    assert not target_dir.exists()


def test_remove_path_if_exists_removes_symlink(tmp_path: Path) -> None:
    target_dir = tmp_path / "run123"
    target_dir.mkdir()

    link = tmp_path / "latest"
    link.symlink_to(target_dir, target_is_directory=True)

    _remove_path_if_exists(link)

    assert not link.exists()
    assert target_dir.exists()
