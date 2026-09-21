"""Stage 0 — resolve and record pinned artefact SHAs into artifacts/manifest.json.

Run this before touching any data (spec §5 Stage 0). It:

* verifies every pinned revision in ``config.yaml`` actually resolves on the
  Hugging Face Hub — a pinned SHA that cannot be resolved is a hard stop, never a
  fall back to ``main`` (CLAUDE.md §4);
* records the config.yaml hash, the current git commit, and a UTC timestamp;
* writes / refreshes ``artifacts/manifest.json``.

The per-file SHA-256 of the frozen release parquet is filled in later by Stage 7;
this stage seeds the manifest with everything knowable before data exists.

Usage:  python -m src.resolve_manifest [--offline]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from src._common import Paths, config_sha256, load_config, utc_now_iso

_HF_API = "https://huggingface.co/api"


def _revision_exists(kind: str, repo_id: str, revision: str, timeout: float = 20.0) -> bool:
    """True iff ``revision`` resolves for ``repo_id`` (kind = 'datasets' | 'models')."""
    url = f"{_HF_API}/{kind}/{repo_id}/revision/{revision}"
    req = urllib.request.Request(url, headers={"User-Agent": "odiaeval-task1/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        raise
    except urllib.error.URLError as exc:  # network down / DNS / TLS
        raise RuntimeError(
            f"could not reach the Hugging Face Hub to verify {repo_id}@{revision}: "
            f"{exc}. Re-run with a working connection, or pass --offline to record "
            f"the pinned SHAs without verification (not acceptable for a release run)."
        ) from exc


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent.parent,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def build_manifest(cfg: dict, *, offline: bool) -> dict:
    ds = cfg["dataset"]
    teacher = cfg["teacher"]

    pinned = [
        ("datasets", ds["repo_id"], ds["revision"], "dataset"),
        ("datasets", ds["repo_id"], ds["parquet_revision"], "dataset_parquet_convert"),
        ("models", teacher["repo_id"], teacher["revision"], "teacher"),
    ]

    resolved: dict[str, dict] = {}
    for kind, repo_id, revision, role in pinned:
        verified = None
        if not offline:
            ok = _revision_exists(kind, repo_id, revision)
            if not ok:
                raise SystemExit(
                    f"FATAL: pinned {role} revision does not resolve: "
                    f"{repo_id}@{revision}. Not falling back to main (CLAUDE.md §4). "
                    f"Fix config.yaml."
                )
            verified = True
        resolved[role] = {
            "repo_id": repo_id,
            "revision": revision,
            "kind": kind,
            "verified_on_hub": verified,
        }

    return {
        "build_timestamp_utc": utc_now_iso(),
        "git_commit": _git_commit(),
        "config_sha256": config_sha256(),
        "seed": cfg["seed"],
        "pinned_artefacts": resolved,
        "dataset_configs": ds["configs"],
        "expected_rows": ds["expected_rows"],
        "release_files": {},  # per-file SHA-256 + row counts filled by Stage 7
        "stage_completed": "0-scaffold",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="record pinned SHAs without Hub verification (dev only, not for release)",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    paths = Paths.from_config(cfg).ensure()
    manifest = build_manifest(cfg, offline=args.offline)

    out = paths.artifacts / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"wrote {out.relative_to(paths.root)}")
    for role, info in manifest["pinned_artefacts"].items():
        tag = "verified" if info["verified_on_hub"] else "UNVERIFIED (offline)"
        print(f"  {role:24s} {info['repo_id']}@{info['revision'][:12]}  [{tag}]")
    print(f"  git_commit               {manifest['git_commit']}")
    print(f"  config_sha256            {manifest['config_sha256'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
