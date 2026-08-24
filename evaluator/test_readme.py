"""The two READMEs must stay usable and stay in step.

A translation drifts silently. The English README gains a flag, a table row, an
environment variable — and the Chinese one keeps describing the version before
it, with nothing failing. Same for links: rename a module and every `.md`
pointing at it becomes a 404 that only a reader discovers.

Three cheap checks, none of which need network or a model:

  1. every relative link in either README resolves to something on disk
  2. both READMEs document the same CLI flags, the same environment variables,
     and the same registered plugins as the code actually has
  3. each points at the other, so a reader can switch language

    pytest evaluator/test_readme.py -v
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_READMES = {"README.md": "English", "README.zh-CN.md": "简体中文"}

pytestmark = pytest.mark.skipif(
    not (_ROOT / "README.md").is_file(),
    reason="repo root not present",
)


def _text(name: str) -> str:
    path = _ROOT / name
    if not path.is_file():
        pytest.fail(f"{name} is missing — the language switcher promises it")
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Relative links resolve
# ---------------------------------------------------------------------------

# `[text](target)` and `src="target"`, skipping anything absolute or anchored.
_MD_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
_HTML_SRC = re.compile(r'(?:src|href)="([^"]+)"')


def _relative_targets(text: str) -> set[str]:
    out = set()
    for pattern in (_MD_LINK, _HTML_SRC):
        for target in pattern.findall(text):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            out.add(target.split("#", 1)[0])
    return {t for t in out if t}


@pytest.mark.parametrize("name", sorted(_READMES))
def test_every_relative_link_resolves(name):
    """CONTRIBUTING says to grep the `.md` files after a rename. This is that
    rule, enforced rather than remembered."""
    missing = sorted(t for t in _relative_targets(_text(name)) if not (_ROOT / t).exists())

    assert not missing, f"{name} points at paths that do not exist: {missing}"


# ---------------------------------------------------------------------------
# 2. Both languages describe the code, not an earlier version of it
# ---------------------------------------------------------------------------


def _runner_flags() -> set[str]:
    source = (_ROOT / "benchmark" / "basic_runner.py").read_text(encoding="utf-8")
    return set(re.findall(r'"(--[a-z][a-z-]+)"', source))


def _documented_flags(text: str) -> set[str]:
    """Flags that appear on an EXPLAINED line, not merely somewhere in the file.

    Substring presence is not enough: every flag used in an example command is
    also "present", so deleting a flag from the reference list while an example
    still mentions it would pass. The reference form is a flag followed by its
    explanation, so that is what gets counted.
    """
    out = set()
    for line in text.splitlines():
        if "#" not in line:
            continue
        before = line.split("#", 1)[0]
        out |= set(re.findall(r"(--[a-z][a-z-]+)", before))
    return out


@pytest.mark.parametrize("name", sorted(_READMES))
def test_documents_every_runner_flag(name):
    """A flag nobody documents is a flag nobody uses."""
    undocumented = sorted(_runner_flags() - _documented_flags(_text(name)))

    assert not undocumented, f"{name} does not document: {undocumented}"


@pytest.mark.parametrize("name", sorted(_READMES))
def test_documents_every_registered_target(name):
    """The plugin table is the only place a reader learns what can be evaluated.

    `mirobody` — the target this repo exists for — was absent from it until
    this test was written, while appearing all over the command examples. So
    the check is scoped to the TABLE ROW: a target named in an example is not
    a target a reader can discover.
    """
    import evaluator.plugin.target_agent  # noqa: F401
    from evaluator.core.interfaces.abstract_target_agent import AbstractTargetAgent

    # The row itself, not the prose row in "Why …" that also names the three
    # agent types — matching on the cell start is what separates them.
    rows = [ln for ln in _text(name).splitlines() if ln.startswith("| **TargetAgent**")]
    assert rows, f"{name} has no TargetAgent row in its plugin table"
    row = rows[0]

    missing = sorted(t for t in AbstractTargetAgent.get_all() if f"`{t}`" not in row)

    assert not missing, f"{name}'s plugin table omits registered targets: {missing}"


@pytest.mark.parametrize("name", sorted(_READMES))
def test_documents_every_environment_variable_the_code_reads(name):
    """`MIROBODY_CONFIG` decides which deployment gets seeded and scored. An
    undocumented one of those is worse than a missing flag: the failure it
    produces names a database, not a setting."""
    read_by_code = set()
    for path in (_ROOT / "evaluator").rglob("*.py"):
        if path.name.startswith("test_"):
            continue
        source = path.read_text(encoding="utf-8")
        read_by_code |= set(re.findall(r'environ(?:\.get)?[\(\[]"(MIROBODY_[A-Z_]+)"', source))

    # Scoped to table rows for the same reason as the plugin check: a variable
    # mentioned in prose is not a variable a reader can look up.
    rows = "\n".join(ln for ln in _text(name).splitlines() if ln.startswith("|"))
    missing = sorted(v for v in read_by_code if f"`{v}`" not in rows)

    assert not missing, f"{name}'s configuration table does not document: {missing}"


def test_env_example_covers_what_the_readme_documents():
    """`.env.example` is the first file a reader copies.

    A variable documented in the README but absent from the template is one the
    reader has to invent from prose — and the ones most worth having in front of
    you are exactly the ones you would not guess, like the path to the
    deployment you are pointing at.
    """
    documented = set()
    rows = [ln for ln in _text("README.md").splitlines() if ln.startswith("|")]
    for row in rows:
        documented |= set(re.findall(r"`([A-Z][A-Z0-9_]{3,})`", row))

    # An ASSIGNMENT line, commented or not — not a mention in the surrounding
    # prose. Every variable this file explains is also "present" in it, so a
    # substring check passes after the line itself is deleted. (That mistake was
    # made three times while writing these checks, in three different ways.)
    template = (_ROOT / ".env.example").read_text(encoding="utf-8")
    assigned = set(re.findall(r"(?m)^\s*#?\s*([A-Z][A-Z0-9_]{3,})=", template))
    missing = sorted(documented - assigned)

    assert not missing, f".env.example has no line for: {missing}"


# ---------------------------------------------------------------------------
# 3. A reader can get from one language to the other
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_READMES))
def test_links_to_the_other_language(name):
    others = [other for other in _READMES if other != name]
    text = _text(name)

    for other in others:
        assert other in text, f"{name} has no link to {other}"


@pytest.mark.parametrize("name", sorted(_READMES))
def test_both_cover_the_same_sections(name):
    """Not the same wording — the same count of top-level sections. A
    translation that quietly stops one section short is the failure mode."""
    counts = {n: len(re.findall(r"(?m)^## ", _text(n))) for n in _READMES}

    assert len(set(counts.values())) == 1, f"section counts differ: {counts}"
