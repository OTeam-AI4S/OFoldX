from __future__ import annotations

from pathlib import Path

import huggingface_hub
import pytest

from ofoldx.hub import MIRROR_REPOSITORY, upload_repository


def test_selective_upload_does_not_delete_unselected_remote_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class FakeHfApi:
        def __init__(self, *, token: str | None) -> None:
            del token

        def create_repo(self, **kwargs: object) -> None:
            del kwargs

        def upload_folder(self, **kwargs: object) -> object:
            calls.append(kwargs)
            return object()

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHfApi)

    upload_repository(tmp_path, "owner/model", allow_patterns=("model.json",))
    upload_repository(tmp_path, "owner/model")

    assert calls[0]["delete_patterns"] is None
    assert calls[1]["delete_patterns"] == list(MIRROR_REPOSITORY)
