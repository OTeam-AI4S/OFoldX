r"""
Hugging Face Hub integration helpers.
"""

from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, suppress
from os import PathLike
from pathlib import Path
from typing import Any, Iterator, cast

from chanfig import Config

from .artifacts import artifact_kind, artifact_task, read_artifact_manifest

MODEL_CARD_NAME = "README.md"
_PROJECT_REPOSITORY = "https://github.com/OTeam-AI4S/OFoldX"
_PROJECT_ISSUES = "https://github.com/OTeam-AI4S/OFoldX/issues"


def resolve_pretrained_path(
    pretrained_name_or_path: str | PathLike[str],
    *,
    repo_type: str = "model",
    revision: str | None = None,
    cache_dir: str | PathLike[str] | None = None,
    local_files_only: bool = False,
    token: str | bool | None = None,
    allow_patterns: Sequence[str] | None = None,
) -> Path:
    r"""
    Return a local directory, downloading a Hub repo snapshot when needed.

    Args:
        pretrained_name_or_path: Local directory path or Hugging Face Hub repository id.
        repo_type: Hub repository type passed to ``snapshot_download``.
        revision: Optional branch, tag, or commit id to resolve.
        cache_dir: Optional Hugging Face cache directory.
        local_files_only: If ``True``, never contact the Hub and resolve only local cache entries.
        token: Optional Hugging Face authentication token.
        allow_patterns: Optional file patterns to include when downloading a Hub snapshot.

    Returns:
        Local directory path containing the requested pretrained artifact.

    Raises:
        ImportError: If a Hub repository is requested but ``huggingface-hub`` is unavailable.

    Examples:
        >>> import tempfile, os
        >>> d = tempfile.mkdtemp()
        >>> resolve_pretrained_path(d) == Path(d)
        True
        >>> resolve_pretrained_path("oteam/protein-model")  # doctest: +SKIP
        PosixPath('/home/user/.cache/huggingface/hub/models--oteam--protein-model/snapshots/...')
    """

    local_path = Path(pretrained_name_or_path).expanduser()
    if local_path.exists():
        return local_path
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:  # pragma: no cover - exercised only without optional dependency.
        raise ImportError("loading from the Hugging Face Hub requires huggingface-hub") from error

    return Path(
        snapshot_download(
            repo_id=str(pretrained_name_or_path),
            repo_type=repo_type,
            revision=revision,
            cache_dir=None if cache_dir is None else str(cache_dir),
            local_files_only=local_files_only,
            token=token,
            allow_patterns=None if allow_patterns is None else list(allow_patterns),
        )
    )


def pretrained_allow_patterns(
    *patterns: str,
    trust_remote_code: bool = False,
) -> tuple[str, ...]:
    r"""
    Return file patterns needed to download a pretrained artifact.

    Args:
        *patterns: Required artifact files.
        trust_remote_code: If ``True``, include Python source files from the artifact repository.

    Returns:
        Tuple of allow-pattern strings for ``huggingface_hub.snapshot_download``. When remote code
        is trusted the ``*.py`` and ``**/*.py`` globs are appended so custom model code is fetched.

    Examples:
        >>> pretrained_allow_patterns("config.json", "*.safetensors")
        ('config.json', '*.safetensors')
        >>> pretrained_allow_patterns("config.json", trust_remote_code=True)
        ('config.json', '*.py', '**/*.py')
    """

    if not trust_remote_code:
        return patterns
    return (*patterns, "*.py", "**/*.py")


@contextmanager
def remote_code_import_path(path: str | PathLike[str], *, trust_remote_code: bool) -> Iterator[None]:
    r"""
    Temporarily add a pretrained artifact directory to ``sys.path`` when remote code is trusted.

    When ``trust_remote_code`` is ``False`` this is a no-op that leaves ``sys.path`` untouched; the
    artifact directory is inserted at the front of ``sys.path`` only for the duration of the block
    and removed again on exit, so custom model modules do not leak into later imports.

    Args:
        path: Local artifact directory.
        trust_remote_code: Whether importing Python code from ``path`` is explicitly allowed.

    Yields:
        ``None`` while imports may resolve modules from ``path``.

    Examples:
        >>> with remote_code_import_path("/tmp/model", trust_remote_code=True):  # doctest: +SKIP
        ...     from modeling_custom import CustomModel
    """

    if not trust_remote_code:
        yield
        return
    code_path = str(Path(path).resolve())
    sys.path.insert(0, code_path)
    importlib.invalidate_caches()
    try:
        yield
    finally:
        with suppress(ValueError):
            sys.path.remove(code_path)
        importlib.invalidate_caches()


def upload_folder_to_hub(
    folder_path: str | PathLike[str],
    *,
    repo_id: str,
    repo_type: str = "model",
    private: bool | None = None,
    token: str | bool | None = None,
    revision: str | None = None,
    commit_message: str | None = None,
    create_pr: bool = False,
    delete_existing: bool = False,
    tag: str | None = None,
    allow_patterns: Sequence[str] | None = None,
    ignore_patterns: Sequence[str] | None = None,
) -> Any:
    r"""
    Create/update a Hub repo with the contents of ``folder_path``.

    Args:
        folder_path: Local directory to upload.
        repo_id: Destination Hugging Face Hub repository id.
        repo_type: Hub repository type passed to ``HfApi``.
        private: Optional privacy setting used when creating the repository.
        token: Optional Hugging Face authentication token.
        revision: Optional target branch for the upload.
        commit_message: Optional commit message for the upload.
        create_pr: If ``True``, upload changes through a Hub pull request.
        delete_existing: If ``True``, delete an existing destination repo before recreating it.
        tag: Optional Hub tag to create after the upload.
        allow_patterns: Optional file patterns to include in the upload.
        ignore_patterns: Optional file patterns to exclude from the upload.

    Returns:
        The object returned by ``huggingface_hub.HfApi.upload_folder`` (the commit info or the
        pull-request URL when ``create_pr`` is set).

    Raises:
        ImportError: If ``huggingface-hub`` is unavailable.

    Examples:
        >>> upload_folder_to_hub("checkpoints/run-1", repo_id="oteam/protein-model")  # doctest: +SKIP
    """

    try:
        from huggingface_hub import HfApi
    except ImportError as error:  # pragma: no cover - exercised only without optional dependency.
        raise ImportError("pushing to the Hugging Face Hub requires huggingface-hub") from error

    api = HfApi()
    if delete_existing:
        api.delete_repo(repo_id=repo_id, repo_type=repo_type, token=token, missing_ok=True)
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, token=token, exist_ok=True)
    if tag is not None and create_pr:
        raise ValueError("cannot create a Hub tag for a pull-request upload; merge the PR and tag the result")
    upload = api.upload_folder(
        folder_path=str(folder_path),
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        commit_message=commit_message,
        token=token,
        create_pr=create_pr,
        allow_patterns=None if allow_patterns is None else list(allow_patterns),
        ignore_patterns=None if ignore_patterns is None else list(ignore_patterns),
    )
    if tag is not None:
        api.create_tag(repo_id=repo_id, repo_type=repo_type, token=token, tag=tag, revision=revision)
    return upload


def publish_artifact_to_hub(
    folder_path: str | PathLike[str],
    *,
    repo_id: str,
    repo_type: str = "model",
    private: bool | None = None,
    token: str | bool | None = None,
    revision: str | None = None,
    commit_message: str | None = None,
    create_pr: bool = False,
    delete_existing: bool = False,
    tag: str | None = None,
    model_card: bool = True,
    model_card_metadata: Mapping[str, object] | None = None,
    widget_examples: Sequence[Mapping[str, object]] | None = None,
    allow_patterns: Sequence[str] | None = None,
    ignore_patterns: Sequence[str] | None = None,
) -> Any:
    r"""
    Prepare and upload a pretrained artifact directory to the Hugging Face Hub.

    This is the artifact-aware publish path: it optionally writes ``README.md`` from the root
    ``manifest.json`` and then delegates the transport step to
    [`upload_folder_to_hub`][ofoldx.hub.upload_folder_to_hub].

    Args:
        folder_path: Local artifact directory to publish.
        repo_id: Destination Hugging Face Hub repository id.
        repo_type: Hub repository type.
        private: Optional privacy setting used when creating the repository.
        token: Optional Hugging Face authentication token.
        revision: Optional target branch for the upload.
        commit_message: Optional commit message for the upload.
        create_pr: If ``True``, upload changes through a Hub pull request.
        delete_existing: If ``True``, delete an existing destination repo before recreating it.
        tag: Optional Hub tag to create after the upload.
        model_card: If ``True``, write/update ``README.md`` from the artifact manifest before upload.
        model_card_metadata: Extra model-card front matter values.
        widget_examples: Optional widget examples merged into a generated card.
        allow_patterns: Optional file patterns to include in the upload.
        ignore_patterns: Optional file patterns to exclude from the upload.

    Returns:
        The object returned by ``huggingface_hub.HfApi.upload_folder``.
    """

    folder = Path(folder_path)
    if model_card:
        write_model_card(
            folder,
            repo_id=repo_id,
            metadata=model_card_metadata,
            widget_examples=widget_examples,
        )
    return upload_folder_to_hub(
        folder,
        repo_id=repo_id,
        repo_type=repo_type,
        private=private,
        token=token,
        revision=revision,
        commit_message=commit_message,
        create_pr=create_pr,
        delete_existing=delete_existing,
        tag=tag,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )


def write_model_card(
    folder_path: str | PathLike[str],
    *,
    repo_id: str | None = None,
    metadata: Mapping[str, object] | None = None,
    widget_examples: Sequence[Mapping[str, object]] | None = None,
) -> Path:
    r"""
    Ensure a saved pretrained artifact has a ``README.md`` model card.

    The card is authored per-model and travels with the artifact (each family ships a self-contained
    ``README.md`` that [`ModelMixin.save_pretrained`][ofoldx.models.mixin.ModelMixin.save_pretrained] copies
    in). When such an authored card is already present, it **is** the model card and is left untouched — the
    single source of truth lives in the family directory, not in this shared module. For a bare artifact with
    no authored card, a compact generic card is generated from the ``manifest.json`` (plus any caller
    ``metadata`` / ``widget_examples`` merged into its front matter).

    Args:
        folder_path: Artifact directory containing ``manifest.json``.
        repo_id: Optional Hub repo id used in the generated title and load snippet.
        metadata: Extra front matter values merged into a generated card (ignored when an authored card exists).
        widget_examples: Optional widget examples for a generated card.

    Returns:
        Path to the ``README.md`` (authored card left in place, or the freshly generated card).

    Raises:
        FileNotFoundError: If the directory is missing ``manifest.json``.

    Examples:
        >>> from tempfile import TemporaryDirectory
        >>> from ofoldx.artifacts import ARTIFACT_MANIFEST_NAME, build_artifact_manifest
        >>> with TemporaryDirectory() as directory:
        ...     build_artifact_manifest(
        ...         kind="model",
        ...         entrypoint="demo.model.DemoModel",
        ...         architecture="demo",
        ...         task="structure_prediction",
        ...     ).json(Path(directory) / ARTIFACT_MANIFEST_NAME)
        ...     path = write_model_card(directory, repo_id="oteam/demo")
        ...     path.name
        'README.md'
    """

    folder = Path(folder_path)
    manifest = read_artifact_manifest(folder)
    card_path = folder / MODEL_CARD_NAME
    frontmatter, existing_body = ("", "")
    if card_path.is_file():
        frontmatter, existing_body = _split_model_card(card_path.read_text())
    if frontmatter.strip():
        # A self-contained authored card is already present -- it IS the model card. Leave it verbatim.
        return card_path
    card_metadata = model_card_metadata(
        manifest,
        repo_id=repo_id,
        metadata=metadata,
        widget_examples=widget_examples,
    )
    card_body = existing_body if existing_body.strip() else _generic_card_body(manifest, repo_id=repo_id)
    card_path.write_text(_format_model_card(card_metadata, card_body))
    return card_path


def _generic_card_body(manifest: Config, *, repo_id: str | None) -> str:
    r"""Return a compact generic card body for a bare artifact that ships no authored card."""

    kind = artifact_kind(manifest)
    architecture = manifest.get("architecture")
    title = repo_id.rsplit("/", 1)[-1] if repo_id else str(architecture or kind)
    detail = f" for the `{architecture}` architecture" if isinstance(architecture, str) and architecture else ""
    snippet = _loader_snippet(kind, _producer_package(manifest), repo_id)
    usage = f"## Usage\n\n```python\n{snippet}\n```\n" if snippet else ""
    return f"# {title}\n\nOFoldX {kind} artifact{detail}.\n\n{usage}"


def model_card_metadata(
    manifest: Config,
    *,
    repo_id: str | None = None,
    metadata: Mapping[str, object] | None = None,
    widget_examples: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    r"""
    Build Hub model-card front matter for an artifact manifest.

    Args:
        manifest: Parsed artifact manifest.
        repo_id: Optional Hub repo id.
        metadata: Extra metadata values. ``tags`` are merged with the generic tags; ``widget: None`` drops the widget.
        widget_examples: Optional widget examples; when omitted no widget is added.

    Returns:
        Plain dictionary suitable for YAML front matter.
    """

    task = artifact_task(manifest)
    kind = artifact_kind(manifest)
    architecture = manifest.get("architecture")
    title = repo_id.rsplit("/", 1)[-1] if repo_id else str(architecture or kind)
    card: dict[str, object] = {
        "library_name": _producer_package(manifest),
        "tags": _dedupe_strings(["biology", kind]),
        "artifact_kind": kind,
    }
    if repo_id is not None:
        card["repo_id"] = repo_id
    if task is not None:
        card["pipeline_tag"] = "other"
        card["task"] = task
    card["model-index"] = [{"name": title, "results": []}]
    if widget_examples:
        card["widget"] = [dict(example) for example in widget_examples]
    for key, value in ({} if metadata is None else dict(metadata)).items():
        if key == "tags":
            generated_tags = cast(Sequence[object], card["tags"])
            card["tags"] = _dedupe_strings([*generated_tags, *_as_sequence(value)])
        elif key == "widget" and value is None:
            card.pop("widget", None)
        else:
            card[key] = value
    return card


def add_hub_upload_arguments(parser: Any) -> None:
    r"""
    Add common Hugging Face Hub upload arguments to a conversion CLI parser.

    Args:
        parser: ``argparse.ArgumentParser``-compatible object.
    """

    group = parser.add_argument_group("Hub upload")
    group.add_argument("--push-to-hub", action="store_true", help="Upload the converted artifact to the Hub")
    group.add_argument("--repo-id", default=None, help="Destination Hub repository id, e.g. oteam/model-name")
    group.add_argument("--private", action="store_true", help="Create the destination repo as private")
    group.add_argument("--token", default=None, help="Hugging Face token")
    group.add_argument("--revision", default=None, help="Target branch for upload_folder")
    group.add_argument("--create-pr", action="store_true", help="Upload through a Hub pull request")
    group.add_argument("--delete-existing", action="store_true", help="Delete an existing repo before upload")
    group.add_argument("--tag", default=None, help="Create this Hub tag after upload")
    group.add_argument(
        "--no-model-card",
        dest="model_card",
        action="store_false",
        default=True,
        help="Skip generated README.md/model-card metadata",
    )


def upload_artifact_from_args(folder_path: str | PathLike[str], args: Any, *, commit_message: str | None = None) -> Any:
    r"""
    Upload a converted artifact using attributes populated by ``add_hub_upload_arguments``.

    Args:
        folder_path: Artifact directory to upload.
        args: Parsed CLI namespace.
        commit_message: Optional fallback commit message.

    Returns:
        ``None`` when ``--push-to-hub`` was not set; otherwise the upload result.
    """

    if not getattr(args, "push_to_hub", False):
        return None
    repo_id = getattr(args, "repo_id", None)
    if not repo_id:
        raise ValueError("--repo-id is required with --push-to-hub")
    return publish_artifact_to_hub(
        folder_path,
        repo_id=repo_id,
        private=True if getattr(args, "private", False) else None,
        token=getattr(args, "token", None),
        revision=getattr(args, "revision", None),
        commit_message=commit_message,
        create_pr=getattr(args, "create_pr", False),
        delete_existing=getattr(args, "delete_existing", False),
        tag=getattr(args, "tag", None),
        model_card=getattr(args, "model_card", True),
    )


def _producer_package(manifest: Config) -> str:
    producer = manifest.get("producer")
    if isinstance(producer, Mapping):
        package = producer.get("package")
        if isinstance(package, str) and package:
            return package
    return "ofoldx"


def _loader_snippet(kind: str, package: str, repo_id: str | None) -> str:
    source = json.dumps(repo_id or "./artifact")
    if kind == "model":
        return (
            f"from {package}.pipelines import Pipeline\n\n"
            f"pipeline = Pipeline.from_pretrained({source})\n"
            f"output = pipeline(...)\n\n"
            f"from {package} import AutoModel\n\n"
            f"model = AutoModel.from_pretrained({source})"
        )
    if kind == "processor":
        return f"from {package} import AutoProcessor\n\nprocessor = AutoProcessor.from_pretrained({source})"
    if kind == "pipeline":
        return f"from {package}.pipelines import Pipeline\n\npipeline = Pipeline.from_pretrained({source})"
    return ""


def _read_artifact_config(
    manifest: Config,
    *,
    folder_path: str | PathLike[str] | None,
) -> Mapping[str, object]:
    if folder_path is None:
        return {}
    files = manifest.get("files")
    config_name = "config.json"
    if isinstance(files, Mapping):
        configured_name = files.get("config")
        if isinstance(configured_name, str) and configured_name:
            config_name = configured_name
    config_path = Path(folder_path) / config_name
    if not config_path.is_file():
        return {}
    try:
        data = json.loads(config_path.read_text())
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, Mapping) else {}


def _split_model_card(content: str) -> tuple[str, str]:
    if not content.startswith("---\n"):
        return "", content
    end = content.find("\n---\n", 4)
    if end == -1:
        return "", content
    return content[4:end], content[end + len("\n---\n") :]


def _format_model_card(metadata: Mapping[str, object], body: str) -> str:
    body = body if body.endswith("\n") else f"{body}\n"
    return "---\n" + "\n".join(_yaml_lines(metadata)) + "\n---\n\n" + body.lstrip("\n")


def _yaml_lines(value: object, *, indent: int = 0) -> list[str]:
    pad = " " * indent
    if isinstance(value, Mapping):
        lines: list[str] = []
        for key, item in value.items():
            if _is_scalar(item):
                lines.append(f"{pad}{key}: {_yaml_scalar(item)}")
            else:
                lines.append(f"{pad}{key}:")
                lines.extend(_yaml_lines(item, indent=indent + 2))
        return lines
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        if not value:
            return [f"{pad}[]"]
        lines = []
        for item in value:
            if _is_scalar(item):
                lines.append(f"{pad}- {_yaml_scalar(item)}")
            elif isinstance(item, Mapping):
                item_lines = _yaml_lines(item, indent=indent + 2)
                if item_lines:
                    first = item_lines[0].lstrip()
                    lines.append(f"{pad}- {first}")
                    lines.extend(item_lines[1:])
                else:
                    lines.append(f"{pad}- {{}}")
            else:
                lines.append(f"{pad}-")
                lines.extend(_yaml_lines(item, indent=indent + 2))
        return lines
    return [f"{pad}{_yaml_scalar(value)}"]


def _is_scalar(value: object) -> bool:
    return value is None or isinstance(value, str | int | float | bool)


def _yaml_scalar(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    return json.dumps(str(value))


def _dedupe_strings(values: Sequence[object]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _as_sequence(value: object) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return value
    return (value,)
