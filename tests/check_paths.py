#!/usr/bin/env python
"""`make check-paths` — no machine-specific absolute path in code, config or docs.

    python tests/check_paths.py            # the gate
    python tests/check_paths.py --list     # what it searched, and what it skipped

WHY THIS IS A GATE AND NOT A STYLE RULE

The literal this looks for is the author's own checkout location. Every
occurrence of it in tracked code is a line that works on exactly one machine, and
that is the failure mode the whole refactor exists to remove: a stranger who
clones the repository gets a `FileNotFoundError` naming a directory that has
never existed for them. Paths now live in `dpcode/conf/paths/default.yaml`, keyed
off `DP_REPO_ROOT` with a repository-relative default, so there is a correct
place to write one down.

THE ALLOWLIST IS NOT A SUPPRESSION LIST

Four kinds of file legitimately contain the literal, and rewriting any of them
would falsify a record rather than fix a defect:

  * committed run artifacts (`results/hydra/**`, `project/CLAM/tmp_eval/**`,
    `project/CLAM/results/**`) — these are records of WHAT WAS RUN, including ten
    tracked `.pt` checkpoints that carry the string inside serialised state and
    cannot be edited at all without invalidating them;
  * historical planning and literature documents (`docs/implementation-research/**`)
    — dated reports, not instructions;
  * `project/CLAM/configs/rna/best_sweep.yaml` — the string appears in a COMMENT
    recording where a sweep result came from;
  * `tests/legacy_wrappers/**` — frozen, byte-identical copies of the
    pre-refactor wrappers. The parity test executes them, so editing one would
    silently redefine what parity is measured against.

So the allowlist is PRINTED whenever the check fires. An allowlist nobody sees is
indistinguishable from a bug, and this one has to stay small enough to read.
"""

from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: The literal. Written as a join so that this file does not itself contain it —
#: a gate that trips on its own source is a gate nobody keeps.
NEEDLE = "/workspace/" + "dp-code"

#: Glob patterns, matched against repository-relative paths. Every entry is a
#: historical record; see the module docstring for why each one is here.
ALLOWLIST: tuple[tuple[str, str], ...] = (
    ("results/hydra/**", "a committed Hydra run directory from 2026-05-01"),
    ("project/CLAM/tmp_eval/**", "committed evaluation outputs of the RNA thread"),
    (
        "project/CLAM/results/**",
        "committed run outputs, including 10 .pt checkpoints that contain the "
        "string inside serialised state",
    ),
    ("docs/implementation-research/**", "dated planning and literature reports"),
    (
        "project/CLAM/configs/rna/best_sweep.yaml",
        "a comment recording where a sweep result came from",
    ),
    (
        "tests/legacy_wrappers/**",
        "frozen pre-refactor wrappers; the parity test executes these, so they "
        "must stay byte-identical",
    ),
)


def tracked_files() -> list[str]:
    """Tracked files, plus untracked ones git would add.

    The second list (`--others --exclude-standard`: untracked and NOT ignored)
    is what makes this gate usable before a commit rather than after one. Without
    it, work in progress — this file's own siblings, the moment they were written
    — would be invisible to the check that exists to keep it clean, and the
    failure would surface only to whoever committed it.
    """
    paths: list[str] = []
    for arguments in (["ls-files"], ["ls-files", "--others", "--exclude-standard"]):
        completed = subprocess.run(
            ["git", "-C", str(REPO), *arguments],
            capture_output=True,
            text=True,
            check=True,
        )
        paths.extend(line for line in completed.stdout.splitlines() if line)
    return sorted(set(paths))


def is_allowlisted(path: str) -> str | None:
    for pattern, reason in ALLOWLIST:
        if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(path, pattern.rstrip("/*") + "/*"):
            return reason
    return None


def offenders() -> tuple[list[tuple[str, int, str]], list[str]]:
    """Return `(violations, skipped)`; violations are `(path, line_no, text)`."""
    violations: list[tuple[str, int, str]] = []
    skipped: list[str] = []
    for relative in tracked_files():
        path = REPO / relative
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary: .pt, .png, .xlsx, .zip
        if NEEDLE not in text:
            continue
        if is_allowlisted(relative):
            skipped.append(relative)
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if NEEDLE in line:
                violations.append((relative, number, line.strip()[:160]))
    return violations, skipped


def print_allowlist() -> None:
    print("Allowlisted (historical records that must not be rewritten):")
    for pattern, reason in ALLOWLIST:
        print(f"  {pattern:<40s} {reason}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--list", action="store_true", help="print the allowlist and the file count, then exit"
    )
    arguments = parser.parse_args(argv)

    if arguments.list:
        print(
            f"searched: {len(tracked_files())} files under {REPO} "
            "(tracked, plus untracked and not ignored)"
        )
        print(f"needle  : {NEEDLE}")
        print_allowlist()
        return 0

    violations, skipped = offenders()
    print(
        f"check-paths: {len(tracked_files())} files searched for {NEEDLE!r} "
        "(tracked, plus untracked and not ignored)"
    )
    if skipped:
        print(f"check-paths: {len(skipped)} allowlisted file(s) contain it and were skipped")

    if not violations:
        print("check-paths: OK — no machine-specific absolute path in tracked code or docs.")
        return 0

    print()
    print(f"check-paths: FAILED — {len(violations)} occurrence(s) in "
          f"{len({v[0] for v in violations})} file(s):")
    for relative, number, line in violations:
        print(f"  {relative}:{number}: {line}")
    print()
    print("Each one is a line that works on exactly one machine. Replace it with a")
    print("`paths.*` key from dpcode/conf/paths/default.yaml, or with a path derived")
    print("from `dpcode.paths.repo_root()` / `Path(__file__)`.")
    print()
    print_allowlist()
    return 1


if __name__ == "__main__":
    sys.exit(main())
