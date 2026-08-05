# Copyright (c) 2014-2026, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the LBANN Research Team (B. Van Essen, et al.) listed in
# the CONTRIBUTORS file. See the top-level LICENSE file for details.
#
# LLNL-CODE-697807.
# All rights reserved.
#
# This file is part of LBANN: Livermore Big Artificial Neural Network
# Toolkit. For details, see http://software.llnl.gov/LBANN or
# https://github.com/LBANN and https://github.com/LBANN/ScaFFold.
#
# SPDX-License-Identifier: (Apache-2.0)

"""Dataset provenance: the commit stamped on a dataset is ScaFFold's.

``meta.yaml``'s ``code_commit``, the published ``<timestamp>__<commit>``
directory name, and the ``dataset_reuse_enforce_commit_id`` gate all key off
one string. Reading it from the *launch* directory made it a property of
wherever the job happened to start -- a site workflow repo, a scratch
directory -- instead of the code that generated the data. Reuse was then gated
on an unrelated repo's churn while real ScaFFold changes went undetected.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from ScaFFold.datagen import get_dataset as gd

LOG = logging.getLogger("test_provenance")

# The ScaFFold source tree: the checkout whose commit must be stamped.
PACKAGE_DIR = Path(gd.__file__).resolve().parent


def _head_of(repo: Path) -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=repo)
        .decode()
        .strip()
    )


def _make_repo(path: Path) -> str:
    """Create a throwaway git repo with one commit; return its short HEAD."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    (path / "README").write_text("an unrelated project\n")
    subprocess.run(["git", "add", "README"], cwd=path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "init",
        ],
        cwd=path,
        check=True,
    )
    return _head_of(path)


def test_commit_is_read_from_the_scaffold_tree_not_the_cwd(tmp_path, monkeypatch):
    """Running from an unrelated repo still stamps ScaFFold's commit."""
    expected = _head_of(PACKAGE_DIR)

    other = tmp_path / "workflow-repo"
    other_head = _make_repo(other)
    assert other_head != expected, "the throwaway repo must differ from ScaFFold"

    monkeypatch.chdir(other)
    assert gd._git_commit_short(LOG) == expected


def test_commit_survives_a_non_repo_working_directory(tmp_path, monkeypatch):
    """A scratch launch directory does not degrade provenance to no-commit-id."""
    expected = _head_of(PACKAGE_DIR)

    scratch = tmp_path / "scratch-cwd"
    scratch.mkdir()
    monkeypatch.chdir(scratch)

    assert gd._git_commit_short(LOG) == expected


def test_non_repo_install_reports_no_commit_id(tmp_path, monkeypatch):
    """An installed (non-git) ScaFFold still degrades gracefully.

    Provenance is best-effort: when the source tree is not a checkout there is
    no commit to record, and reuse simply is not gated on one.

    "Not a checkout" has to be made true of the *whole path*, not just the leaf:
    git walks upwards until it finds a repository, so with ``--basetemp`` inside
    a ScaFFold checkout this directory inherits that checkout's HEAD and the
    test fails on where it was run rather than on what it tests. The ceiling
    stops the walk at ``tmp_path``.
    """
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    not_a_repo = tmp_path / "site-packages" / "ScaFFold" / "datagen"
    not_a_repo.mkdir(parents=True)

    assert gd._git_commit_short(LOG, source_dir=not_a_repo) == "no-commit-id"


def test_missing_source_dir_reports_no_commit_id(tmp_path):
    """A source directory that does not exist is handled, not raised."""
    assert gd._git_commit_short(LOG, source_dir=tmp_path / "gone") == "no-commit-id"
