"""Preview files are restricted to server-owned upload and sample directories."""

from pathlib import Path

import pytest

import preview_handler


def test_uploaded_file_inside_gradio_folder_is_allowed(tmp_path, monkeypatch):
    upload_root = tmp_path / "gradio"
    uploaded = upload_root / "session" / "claim.pdf"
    uploaded.parent.mkdir(parents=True)
    uploaded.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(preview_handler, "get_upload_folder", lambda: str(upload_root))

    assert preview_handler._resolve_preview_path(uploaded) == uploaded.resolve()


def test_sample_file_inside_sample_root_is_allowed(tmp_path, monkeypatch):
    sample = tmp_path / "sample" / "claims" / "claim.pdf"
    sample.parent.mkdir(parents=True)
    sample.write_bytes(b"%PDF-1.4")
    monkeypatch.chdir(tmp_path)

    assert preview_handler._resolve_preview_path(sample) == sample.resolve()


def test_relative_sample_path_inside_sample_root_is_allowed(tmp_path, monkeypatch):
    sample = tmp_path / "sample" / "claims" / "claim.pdf"
    sample.parent.mkdir(parents=True)
    sample.write_bytes(b"%PDF-1.4")
    monkeypatch.chdir(tmp_path)

    assert (
        preview_handler._resolve_preview_path(Path("sample/claims/claim.pdf"))
        == sample.resolve()
    )


def test_preview_path_outside_allowed_roots_is_rejected(tmp_path, monkeypatch):
    upload_root = tmp_path / "gradio"
    upload_root.mkdir()
    outside = tmp_path / "private.pdf"
    outside.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(preview_handler, "get_upload_folder", lambda: str(upload_root))
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="outside"):
        preview_handler._resolve_preview_path(outside)


def test_symlink_cannot_escape_an_allowed_root(tmp_path, monkeypatch):
    upload_root = tmp_path / "gradio"
    upload_root.mkdir()
    outside = tmp_path / "private.pdf"
    outside.write_bytes(b"%PDF-1.4")
    linked = upload_root / "claim.pdf"
    linked.symlink_to(outside)
    monkeypatch.setattr(preview_handler, "get_upload_folder", lambda: str(upload_root))
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="outside"):
        preview_handler._resolve_preview_path(linked)


def test_pdf_info_escapes_the_uploaded_filename():
    info_html = preview_handler.create_pdf_info_html(
        Path("/tmp/<script>alert(1)<script>.pdf"),
        current_page=0,
        total_pages=1,
    )

    assert "<script>" not in info_html
    assert "&lt;script&gt;" in info_html
