r"""
HMMER tool wrappers for MSA generation.

Ports the AlphaFold3 tool-driven MSA path (``data/tools/{jackhmmer,nhmmer,hmmbuild,hmmsearch,hmmalign}.py``
and the ``data/pipeline.py`` orchestration) so a query sequence can be aligned against genetic databases
into an a3m alignment. This is the generation counterpart to the precomputed-a3m path in
[`encode`][ofoldx.processors.msa.encode] (which encodes an existing alignment).

EXTERNAL DEPENDENCIES (REQUIRED for this path, NOT bundled with OFoldX):
  * HMMER binaries -- ``jackhmmer``, ``nhmmer``, ``hmmbuild``, ``hmmsearch``, ``hmmalign`` (install the
    HMMER suite, https://hmmer.org).
  * Genetic sequence databases (FASTA) -- e.g. UniRef90, MGnify, BFD/small-BFD, UniProt for ofoldx and
    an RNA database for nucleic acids. These are tens-to-hundreds of GB and must be downloaded separately.

Each wrapper resolves and validates its binary at construction and raises [`MsaToolError`][] with an
actionable message if the binary is missing, so a caller learns the dependency is unmet rather than
seeing an opaque failure. No MSA output is ever fabricated: without the binaries and databases these
wrappers cannot and do not produce alignments.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from concurrent import futures
from typing import IO, Protocol


class MsaToolError(RuntimeError):
    r"""
    Raised when an MSA tool cannot run (missing binary, missing database, or a non-zero exit).
    """


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class MsaToolResult:
    r"""
    Result of an MSA-tool query.

    Attributes:
        target_sequence: Query sequence the tool was run on.
        e_value: Reporting E-value used to filter hits.
        a3m: Alignment in a3m format.

    Examples:
        >>> result = MsaToolResult(target_sequence="MKT", e_value=1e-3, a3m=">query\nMKT\n")
        >>> result.target_sequence
        'MKT'
    """

    target_sequence: str
    e_value: float
    a3m: str


class MsaTool(Protocol):
    r"""
    Interface for MSA tools that align a query sequence against a database.

    Implementations run an external aligner and return the query's a3m alignment via
    [`query`][ofoldx.processors.msa.search.MsaTool.query].
    """

    def query(self, target_sequence: str) -> MsaToolResult:
        r"""
        Align a query sequence and return its a3m alignment.

        Args:
            target_sequence: Query sequence.

        Returns:
            The tool's alignment result.

        Examples:
            >>> result = tool.query("MKTAYIAKQR")  # doctest: +SKIP
        """

        ...


def resolve_binary(binary_path: str, name: str) -> str:
    r"""
    Return an existing, executable binary path or raise a clear dependency error.

    Args:
        binary_path: Configured path or bare command name for the tool binary.
        name: Human-readable tool name for the error message.

    Returns:
        Resolved absolute path to the binary.

    Raises:
        MsaToolError: If the binary cannot be found or is not executable.

    Examples:
        >>> resolve_binary("jackhmmer", "jackhmmer")  # doctest: +SKIP
        '/usr/local/bin/jackhmmer'
    """

    resolved = shutil.which(binary_path) if os.path.sep not in binary_path else binary_path
    if not resolved or not os.path.exists(resolved) or not os.access(resolved, os.X_OK):
        raise MsaToolError(
            f"{name} binary not found at {binary_path!r}. Install the HMMER suite (https://hmmer.org) "
            f"and pass its path; MSA generation cannot run without it."
        )
    return resolved


def run_command(cmd: Sequence[str], *, name: str) -> subprocess.CompletedProcess[str]:
    r"""
    Run a subprocess command, raising a clear error on failure.

    Args:
        cmd: Command and arguments.
        name: Tool name for error messages.

    Returns:
        The completed process.

    Raises:
        MsaToolError: If the command exits non-zero or the binary is missing.

    Examples:
        >>> run_command(["hmmbuild", "profile.hmm", "input.a3m"], name="hmmbuild")  # doctest: +SKIP
    """

    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise MsaToolError(f"{name} binary {cmd[0]!r} not found: {error}") from error
    except subprocess.CalledProcessError as error:
        raise MsaToolError(f"{name} failed (exit {error.returncode}): {error.stderr}") from error


def write_query_fasta(sequence: str, path: str) -> None:
    r"""
    Write a single-sequence query FASTA file.

    Args:
        sequence: Query sequence.
        path: Output FASTA path.

    Examples:
        >>> import tempfile, os
        >>> path = os.path.join(tempfile.mkdtemp(), "query.fasta")
        >>> write_query_fasta("MKT", path)
        >>> open(path).read()
        '>query\nMKT\n'
    """

    with open(path, "w") as handle:
        handle.write(f">query\n{sequence}\n")


class Jackhmmer:
    r"""
    Wrapper around the ``jackhmmer`` binary for iterative protein-sequence search.

    Requires the ``jackhmmer`` binary and a FASTA sequence database (e.g. UniRef90 or MGnify), neither
    of which is bundled. The binary is validated at construction.

    Args:
        binary_path: Path or command name for ``jackhmmer``.
        database_path: Path to a FASTA sequence database.
        n_cpu: CPU count passed to ``jackhmmer``.
        n_iter: Number of search iterations.
        e_value: Reporting and inclusion E-value.
        z_value: Optional effective database size for E-value scaling.
        max_sequences: Maximum alignment depth to keep.
        filter_f1: MSV/bias pre-filter threshold.
        filter_f2: Viterbi pre-filter threshold.
        filter_f3: Forward pre-filter threshold.

    Raises:
        MsaToolError: If the ``jackhmmer`` binary is unavailable.

    Examples:
        >>> tool = Jackhmmer(database_path="uniref90.fasta")  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        binary_path: str = "jackhmmer",
        database_path: str,
        n_cpu: int = 8,
        n_iter: int = 3,
        e_value: float = 1e-3,
        z_value: float | None = None,
        max_sequences: int = 5000,
        filter_f1: float = 5e-4,
        filter_f2: float = 5e-5,
        filter_f3: float = 5e-7,
    ) -> None:
        self._binary_path = resolve_binary(binary_path, "jackhmmer")
        self._database_path = database_path
        self._n_cpu = n_cpu
        self._n_iter = n_iter
        self._e_value = e_value
        self._z_value = z_value
        self._max_sequences = max_sequences
        self._filter_f1 = filter_f1
        self._filter_f2 = filter_f2
        self._filter_f3 = filter_f3

    def query(self, target_sequence: str) -> MsaToolResult:
        r"""
        Search the database with ``jackhmmer`` and return the a3m alignment.

        Args:
            target_sequence: Query protein sequence.

        Returns:
            The alignment result.

        Raises:
            MsaToolError: If the database is missing or ``jackhmmer`` fails.

        Examples:
            >>> tool = Jackhmmer(database_path="uniref90.fasta")  # doctest: +SKIP
            >>> result = tool.query("MKTAYIAKQR")  # doctest: +SKIP
        """

        if not os.path.exists(self._database_path):
            raise MsaToolError(
                f"jackhmmer database not found at {self._database_path!r}. Download the genetic database "
                f"(e.g. UniRef90) separately; MSA generation cannot run without it."
            )
        with tempfile.TemporaryDirectory() as tmp_dir:
            fasta_path = os.path.join(tmp_dir, "query.fasta")
            sto_path = os.path.join(tmp_dir, "output.sto")
            write_query_fasta(target_sequence, fasta_path)
            cmd = [
                self._binary_path,
                *("-o", "/dev/null"),
                *("-A", sto_path),
                "--noali",
                *("--F1", str(self._filter_f1)),
                *("--F2", str(self._filter_f2)),
                *("--F3", str(self._filter_f3)),
                *("--cpu", str(self._n_cpu)),
                *("-N", str(self._n_iter)),
                *("-E", str(self._e_value)),
                *("--incE", str(self._e_value)),
                *(("-Z", str(self._z_value)) if self._z_value is not None else ()),
                fasta_path,
                self._database_path,
            ]
            run_command(cmd, name="jackhmmer")
            with open(sto_path) as handle:
                a3m = convert_stockholm_to_a3m(handle, max_sequences=self._max_sequences)
        return MsaToolResult(target_sequence=target_sequence, a3m=a3m, e_value=self._e_value)


class Nhmmer:
    r"""
    Wrapper around the ``nhmmer`` binary for nucleotide-sequence search.

    Requires the ``nhmmer``, ``hmmalign``, and ``hmmbuild`` binaries plus a nucleotide FASTA database,
    none of which is bundled.

    Args:
        binary_path: Path or command name for ``nhmmer``.
        hmmalign_binary_path: Path or command name for ``hmmalign``.
        hmmbuild_binary_path: Path or command name for ``hmmbuild``.
        database_path: Path to a nucleotide FASTA database.
        n_cpu: CPU count passed to ``nhmmer``.
        e_value: Reporting E-value.
        max_sequences: Maximum alignment depth to keep.
        alphabet: Sequence alphabet (``"rna"`` or ``"dna"``).

    Raises:
        MsaToolError: If any required binary is unavailable.

    Examples:
        >>> tool = Nhmmer(database_path="rnacentral.fasta")  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        binary_path: str = "nhmmer",
        hmmalign_binary_path: str = "hmmalign",
        hmmbuild_binary_path: str = "hmmbuild",
        database_path: str,
        n_cpu: int = 8,
        e_value: float = 1e-3,
        max_sequences: int = 5000,
        alphabet: str = "rna",
    ) -> None:
        self._binary_path = resolve_binary(binary_path, "nhmmer")
        self._hmmalign_binary_path = resolve_binary(hmmalign_binary_path, "hmmalign")
        self._hmmbuild_binary_path = resolve_binary(hmmbuild_binary_path, "hmmbuild")
        self._database_path = database_path
        self._n_cpu = n_cpu
        self._e_value = e_value
        self._max_sequences = max_sequences
        self._alphabet = alphabet

    def query(self, target_sequence: str) -> MsaToolResult:
        r"""
        Search the nucleotide database with ``nhmmer`` and return the a3m alignment.

        Args:
            target_sequence: Query nucleotide sequence.

        Returns:
            The alignment result.

        Raises:
            MsaToolError: If the database is missing or ``nhmmer`` fails.

        Examples:
            >>> tool = Nhmmer(database_path="rnacentral.fasta")  # doctest: +SKIP
            >>> result = tool.query("GGGCUUAAGG")  # doctest: +SKIP
        """

        if not os.path.exists(self._database_path):
            raise MsaToolError(
                f"nhmmer database not found at {self._database_path!r}. Download the nucleotide database "
                f"separately; MSA generation cannot run without it."
            )
        with tempfile.TemporaryDirectory() as tmp_dir:
            fasta_path = os.path.join(tmp_dir, "query.fasta")
            sto_path = os.path.join(tmp_dir, "output.sto")
            write_query_fasta(target_sequence, fasta_path)
            cmd = [
                self._binary_path,
                *("-o", "/dev/null"),
                *("-A", sto_path),
                "--noali",
                *("--cpu", str(self._n_cpu)),
                *("-E", str(self._e_value)),
                f"--{self._alphabet}",
                fasta_path,
                self._database_path,
            ]
            run_command(cmd, name="nhmmer")
            with open(sto_path) as handle:
                a3m = convert_stockholm_to_a3m(handle, max_sequences=self._max_sequences)
        return MsaToolResult(target_sequence=target_sequence, a3m=a3m, e_value=self._e_value)


class Hmmbuild:
    r"""
    Wrapper around the ``hmmbuild`` binary, building an HMM profile from an alignment.

    Args:
        binary_path: Path or command name for ``hmmbuild``.
        alphabet: Optional alphabet assertion (``"amino"``, ``"rna"``, ``"dna"``).

    Raises:
        MsaToolError: If the ``hmmbuild`` binary is unavailable.

    Examples:
        >>> hmmbuild = Hmmbuild(alphabet="amino")  # doctest: +SKIP
    """

    def __init__(self, *, binary_path: str = "hmmbuild", alphabet: str | None = None) -> None:
        self._binary_path = resolve_binary(binary_path, "hmmbuild")
        self._alphabet = alphabet

    def build_profile_from_a3m(self, a3m: str) -> str:
        r"""
        Build an HMM profile from an a3m alignment.

        Args:
            a3m: Alignment in a3m format.

        Returns:
            HMM profile text.

        Raises:
            MsaToolError: If ``hmmbuild`` fails.

        Examples:
            >>> profile = Hmmbuild(alphabet="amino").build_profile_from_a3m(a3m)  # doctest: +SKIP
        """

        with tempfile.TemporaryDirectory() as tmp_dir:
            a3m_path = os.path.join(tmp_dir, "input.a3m")
            hmm_path = os.path.join(tmp_dir, "profile.hmm")
            with open(a3m_path, "w") as handle:
                handle.write(a3m)
            cmd = [
                self._binary_path,
                "--hand",
                *(("--amino",) if self._alphabet == "amino" else ()),
                *(("--rna",) if self._alphabet == "rna" else ()),
                *(("--dna",) if self._alphabet == "dna" else ()),
                hmm_path,
                a3m_path,
            ]
            run_command(cmd, name="hmmbuild")
            with open(hmm_path) as handle:
                return handle.read()


class Hmmalign:
    r"""
    Wrapper around the ``hmmalign`` binary, aligning sequences to an HMM profile.

    Args:
        binary_path: Path or command name for ``hmmalign``.

    Raises:
        MsaToolError: If the ``hmmalign`` binary is unavailable.

    Examples:
        >>> hmmalign = Hmmalign()  # doctest: +SKIP
    """

    def __init__(self, *, binary_path: str = "hmmalign") -> None:
        self._binary_path = resolve_binary(binary_path, "hmmalign")

    def align_sequences(self, sequences: Sequence[str], profile: str) -> str:
        r"""
        Align sequences to an HMM profile, returning Stockholm output.

        Args:
            sequences: Sequences to align.
            profile: HMM profile text.

        Returns:
            Stockholm-format alignment.

        Raises:
            MsaToolError: If ``hmmalign`` fails.

        Examples:
            >>> stockholm = Hmmalign().align_sequences(["MKTAY", "MKSAY"], profile)  # doctest: +SKIP
        """

        with tempfile.TemporaryDirectory() as tmp_dir:
            profile_path = os.path.join(tmp_dir, "profile.hmm")
            fasta_path = os.path.join(tmp_dir, "input.fasta")
            with open(profile_path, "w") as handle:
                handle.write(profile)
            with open(fasta_path, "w") as handle:
                handle.write("".join(f">seq{index}\n{sequence}\n" for index, sequence in enumerate(sequences)))
            result = run_command(
                [self._binary_path, "--outformat", "Stockholm", profile_path, fasta_path], name="hmmalign"
            )
            return result.stdout


class Hmmsearch:
    r"""
    Wrapper around the ``hmmsearch`` binary, searching a database with an HMM profile.

    Used by the template pipeline to find template hits for a query MSA. Requires the ``hmmsearch`` and
    ``hmmbuild`` binaries plus a template sequence database.

    Args:
        binary_path: Path or command name for ``hmmsearch``.
        hmmbuild_binary_path: Path or command name for ``hmmbuild``.
        database_path: Path to a template sequence FASTA database.
        e_value: Reporting E-value.

    Raises:
        MsaToolError: If a required binary is unavailable.

    Examples:
        >>> hmmsearch = Hmmsearch(database_path="pdb_seqres.fasta")  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        binary_path: str = "hmmsearch",
        hmmbuild_binary_path: str = "hmmbuild",
        database_path: str,
        e_value: float = 1e-3,
    ) -> None:
        self._binary_path = resolve_binary(binary_path, "hmmsearch")
        self._hmmbuild = Hmmbuild(binary_path=hmmbuild_binary_path, alphabet="amino")
        self._database_path = database_path
        self._e_value = e_value

    def query(self, a3m: str) -> str:
        r"""
        Build an HMM from an a3m alignment and search the database, returning Stockholm hits.

        Args:
            a3m: Query alignment in a3m format.

        Returns:
            Stockholm-format hits.

        Raises:
            MsaToolError: If the database is missing or ``hmmsearch`` fails.

        Examples:
            >>> hits = Hmmsearch(database_path="pdb_seqres.fasta").query(query_a3m)  # doctest: +SKIP
        """

        if not os.path.exists(self._database_path):
            raise MsaToolError(
                f"hmmsearch template database not found at {self._database_path!r}. Download the PDB "
                f"sequence database separately; template search cannot run without it."
            )
        profile = self._hmmbuild.build_profile_from_a3m(a3m)
        with tempfile.TemporaryDirectory() as tmp_dir:
            profile_path = os.path.join(tmp_dir, "profile.hmm")
            out_path = os.path.join(tmp_dir, "hits.sto")
            with open(profile_path, "w") as handle:
                handle.write(profile)
            cmd = [
                self._binary_path,
                "--noali",
                *("-E", str(self._e_value)),
                *("-A", out_path),
                profile_path,
                self._database_path,
            ]
            run_command(cmd, name="hmmsearch")
            with open(out_path) as handle:
                return handle.read()


def run_msa_tools(
    target_sequence: str, tools: Sequence[MsaTool], *, max_workers: int | None = None
) -> list[MsaToolResult]:
    r"""
    Run several MSA tools on one query in parallel and return their results.

    Mirrors the upstream pipeline's fan-out over per-database tools (e.g. UniRef90 + MGnify + BFD +
    UniProt); the caller deduplicates and merges the resulting alignments.

    Args:
        target_sequence: Query sequence.
        tools: Configured MSA tools to run.
        max_workers: Optional thread-pool size; defaults to the number of tools.

    Returns:
        One [`MsaToolResult`][] per tool, in input order.

    Examples:
        >>> tools = [Jackhmmer(database_path="uniref90.fasta"),
        ...          Jackhmmer(database_path="mgnify.fasta")]  # doctest: +SKIP
        >>> results = run_msa_tools("MKTAYIAKQR", tools)  # doctest: +SKIP
    """

    with futures.ThreadPoolExecutor(max_workers=max_workers or len(tools) or 1) as executor:
        return list(executor.map(lambda tool: tool.query(target_sequence), tools))


def convert_stockholm_to_a3m(stockholm: IO[str], max_sequences: int | None = None) -> str:
    r"""
    Convert a Stockholm-format alignment to a3m.

    Faithful port of upstream ``parsers.convert_stockholm_to_a3m``: concatenates the wrapped Stockholm
    blocks, aligns each sequence to the gapless query (dropping insert-state ``.`` columns and
    lowercasing residues aligned to query gaps), and emits a3m records.

    Args:
        stockholm: Open text stream of a Stockholm alignment.
        max_sequences: Optional cap on the number of sequences kept.

    Returns:
        The a3m alignment text (empty when the input has no sequences).

    Examples:
        >>> import io
        >>> stockholm = "# STOCKHOLM 1.0\nquery ABCDE\nseq1  A-CDE\n//\n"
        >>> convert_stockholm_to_a3m(io.StringIO(stockholm)).splitlines()
        ['>query', 'ABCDE', '>seq1', 'A-CDE']
    """

    sequences: dict[str, str] = {}
    for line in stockholm:
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "//")):
            continue
        name, _, aligned = stripped.partition(" ")
        aligned = aligned.strip()
        if name not in sequences and max_sequences is not None and len(sequences) >= max_sequences:
            continue
        sequences[name] = sequences.get(name, "") + aligned
    if not sequences:
        return ""

    query_sequence = next(iter(sequences.values()))
    keep_columns = {index for index, residue in enumerate(query_sequence) if residue not in {"-", "."}}
    chunks: list[str] = []
    for name, sequence in sequences.items():
        a3m_residues: list[str] = []
        for index, residue in enumerate(sequence):
            if index in keep_columns:
                a3m_residues.append(residue.upper())
            elif residue not in {"-", "."}:
                a3m_residues.append(residue.lower())
        chunks.append(f">{name}\n{''.join(a3m_residues)}")
    return "\n".join(chunks)
