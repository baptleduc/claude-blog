"""Mechanical security and quality guardrails for the claude-blog repo.

These tests enforce four invariants that complement the rules documented in
CLAUDE.md. Rules in prose can drift; assertions in pytest cannot. If any of
these tests fail in CI, a contributor (human or agent) has regressed a
project-wide invariant and must fix the underlying file before merging.

Invariants enforced:

1. No agent grants the ``Bash`` tool in its YAML frontmatter ``tools`` list.
2. No SKILL.md uses the unsupported ``allowed-tools`` frontmatter key.
3. Every skill has a unique ``name`` (no duplicate command routes).
4. ``scripts/sync_flow.py``, when present, contains the required security
   primitives (host allowlist, size cap, ``--dry-run``, ``--ref``, lock file,
   license-header injection, path-traversal guard).

Stdlib + pytest only. No network, no writes outside ``tmp_path``.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = REPO_ROOT / "agents"
SKILLS_DIR = REPO_ROOT / "skills"
SYNC_FLOW_PATH = REPO_ROOT / "scripts" / "sync_flow.py"


# ---------------------------------------------------------------------------
# Minimal stdlib-only frontmatter parser
# ---------------------------------------------------------------------------
#
# We intentionally avoid PyYAML: it is not a dependency in pyproject.toml and
# the brief forbids new dependencies. The parser is deliberately small and
# only handles the shapes that actually appear in agent and SKILL.md files:
#
#   key: value
#   key: "quoted value"
#   key: >          (folded scalar continued on subsequent indented lines)
#   key:            (followed by a YAML-style list)
#     - item
#     - "item"
#
# Anything more exotic would itself be a code smell in a SKILL.md / agent file
# and is out of scope for these guardrails.

_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(?P<body>.*?)\n---\s*(?:\n|$)",
    re.DOTALL,
)


def _split_frontmatter(text: str) -> str | None:
    """Return the raw YAML body between the leading ``---`` markers.

    Returns ``None`` if the file does not start with a frontmatter block.
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return None
    return match.group("body")


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        return value[1:-1]
    return value


def parse_frontmatter(text: str) -> dict | None:
    """Parse the leading YAML frontmatter into a dict.

    Returns ``None`` when no frontmatter block is present so callers can
    distinguish "missing" from "empty".
    """
    body = _split_frontmatter(text)
    if body is None:
        return None

    result: dict = {}
    lines = body.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        # Skip blank lines and comments at the top level.
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        # Only top-level keys (no leading whitespace) start a new entry.
        if raw[:1] in (" ", "\t"):
            i += 1
            continue

        if ":" not in raw:
            i += 1
            continue

        key, _, rest = raw.partition(":")
        key = key.strip()
        rest = rest.strip()

        # Folded scalar: collect indented continuation lines as a string.
        if rest in (">", "|", ">-", "|-"):
            collected: list[str] = []
            i += 1
            while i < len(lines) and (
                lines[i].startswith((" ", "\t")) or not lines[i].strip()
            ):
                collected.append(lines[i].strip())
                i += 1
            result[key] = " ".join(part for part in collected if part)
            continue

        # YAML-style list on subsequent indented lines.
        if rest == "":
            items: list[str] = []
            i += 1
            while i < len(lines) and (
                lines[i].startswith((" ", "\t")) or not lines[i].strip()
            ):
                line = lines[i]
                line_stripped = line.strip()
                if line_stripped.startswith("- "):
                    items.append(_strip_quotes(line_stripped[2:].strip()))
                elif line_stripped == "-":
                    items.append("")
                # Ignore nested mapping continuations.
                i += 1
            if items:
                result[key] = items
            else:
                # Empty value with no list. Record as None rather than fabricate.
                result[key] = None
            continue

        # Inline list: key: [a, b, c]
        if rest.startswith("[") and rest.endswith("]"):
            inner = rest[1:-1].strip()
            if inner:
                result[key] = [_strip_quotes(p.strip()) for p in inner.split(",")]
            else:
                result[key] = []
            i += 1
            continue

        # Plain scalar.
        result[key] = _strip_quotes(rest)
        i += 1

    return result


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Test 1: No agent grants the Bash tool
# ---------------------------------------------------------------------------


def test_no_bash_tool_in_any_agent_frontmatter() -> None:
    """No ``agents/*.md`` may list ``Bash`` in its ``tools`` frontmatter field.

    Agents that omit ``tools`` entirely receive the default tool surface,
    which is a separate concern handled elsewhere. Files without parseable
    frontmatter are reported as a pytest warning rather than a failure so
    drafts and READMEs do not break CI.
    """
    if not AGENTS_DIR.is_dir():
        pytest.skip(f"agents directory missing at {AGENTS_DIR}")

    agent_files = sorted(AGENTS_DIR.glob("*.md"))
    assert agent_files, f"No agent .md files found under {AGENTS_DIR}"

    offenders: list[str] = []
    for path in agent_files:
        text = _read(path)
        fm = parse_frontmatter(text)
        if fm is None:
            warnings.warn(
                f"Agent {path.relative_to(REPO_ROOT)} has no parseable "
                "frontmatter; skipping Bash check for this file.",
                stacklevel=1,
            )
            continue

        tools = fm.get("tools")
        if tools is None:
            # No tools field => default tool access. Out of scope.
            continue

        # Normalise to a list of strings for inspection.
        if isinstance(tools, str):
            tool_items = [t.strip() for t in tools.split(",")]
        elif isinstance(tools, list):
            tool_items = [str(t).strip() for t in tools]
        else:
            tool_items = []

        # Strip wrapping quotes that may survive if YAML had `"Bash"` or `'Bash'`.
        normalised = [_strip_quotes(t) for t in tool_items]

        if "Bash" in normalised:
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert not offenders, (
        "Found agents granting the `Bash` tool in their frontmatter. "
        "Agents must not have shell access. Remove `Bash` from the `tools:` "
        "list in:\n  - " + "\n  - ".join(offenders)
    )


# ---------------------------------------------------------------------------
# Test 2: No SKILL.md uses the invalid `allowed-tools` field
# ---------------------------------------------------------------------------


def test_no_allowed_tools_field_in_skills() -> None:
    """``allowed-tools`` is not a valid Claude Code SKILL.md frontmatter key.

    Per CLAUDE.md, valid fields are: name, description, user-invokable,
    argument-hint, compatibility, license, metadata, disable-model-invocation.
    Setting ``allowed-tools`` silently does nothing and signals confusion
    with the Claude Code agent / settings schema.
    """
    if not SKILLS_DIR.is_dir():
        pytest.skip(f"skills directory missing at {SKILLS_DIR}")

    skill_files = sorted(SKILLS_DIR.rglob("SKILL.md"))
    assert skill_files, f"No SKILL.md files found under {SKILLS_DIR}"

    offenders: list[str] = []
    for path in skill_files:
        text = _read(path)
        fm = parse_frontmatter(text)
        if fm is None:
            warnings.warn(
                f"Skill {path.relative_to(REPO_ROOT)} has no parseable "
                "frontmatter; cannot check allowed-tools.",
                stacklevel=1,
            )
            continue

        if "allowed-tools" in fm:
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert not offenders, (
        "Found SKILL.md files using the invalid `allowed-tools` frontmatter "
        "key. This field is not part of the Claude Code SKILL.md spec and is "
        "silently ignored. Remove it from:\n  - " + "\n  - ".join(offenders)
    )


# ---------------------------------------------------------------------------
# Test 3: Skill names are globally unique
# ---------------------------------------------------------------------------


def test_unique_skill_names_and_command_routes() -> None:
    """Every skill must declare a unique ``name`` value.

    Duplicate names collide on the slash-command route surface and one of
    the skills will be unreachable. ``argument-hint`` collisions are
    permitted (multiple skills can share the same hint shape) and are not
    a failure here, but duplicate ``name`` values always are.
    """
    if not SKILLS_DIR.is_dir():
        pytest.skip(f"skills directory missing at {SKILLS_DIR}")

    skill_files = sorted(SKILLS_DIR.rglob("SKILL.md"))
    assert skill_files, f"No SKILL.md files found under {SKILLS_DIR}"

    name_to_paths: dict[str, list[str]] = {}
    missing_name: list[str] = []

    for path in skill_files:
        text = _read(path)
        fm = parse_frontmatter(text)
        if fm is None:
            warnings.warn(
                f"Skill {path.relative_to(REPO_ROOT)} has no parseable "
                "frontmatter; cannot check name uniqueness.",
                stacklevel=1,
            )
            continue

        name = fm.get("name")
        if not name or not isinstance(name, str):
            missing_name.append(str(path.relative_to(REPO_ROOT)))
            continue

        name_to_paths.setdefault(name, []).append(str(path.relative_to(REPO_ROOT)))

    duplicate_messages = [
        f"Duplicate skill name '{name}' found at " + " and ".join(paths)
        for name, paths in sorted(name_to_paths.items())
        if len(paths) > 1
    ]

    failure_lines: list[str] = []
    if duplicate_messages:
        failure_lines.append("Skill name collisions:")
        failure_lines.extend(f"  - {msg}" for msg in duplicate_messages)
    if missing_name:
        failure_lines.append(
            "SKILL.md files missing a `name:` frontmatter field:"
        )
        failure_lines.extend(f"  - {p}" for p in missing_name)

    assert not failure_lines, "\n".join(failure_lines)


# ---------------------------------------------------------------------------
# Test 4: sync_flow.py security invariants
# ---------------------------------------------------------------------------


def _check_pattern(source: str, *patterns: str) -> bool:
    """Return True when at least one of ``patterns`` appears in ``source``."""
    return any(p in source for p in patterns)


def test_sync_flow_security_invariants() -> None:
    """``scripts/sync_flow.py`` must encode key security primitives in source.

    These are checked by reading the script as text rather than executing it,
    so the test never makes network calls and never spawns the script. If
    the script does not yet exist, the test skips so it can land before the
    sync flow does.
    """
    if not SYNC_FLOW_PATH.exists():
        pytest.skip("sync_flow.py not yet present")

    source = _read(SYNC_FLOW_PATH)

    # Each entry: human label -> tuple of acceptable substrings (any match wins).
    invariants: dict[str, tuple[str, ...]] = {
        "host allowlist references api.github.com": ("api.github.com",),
        "explicit host validator (function or constant)": (
            "_validate_github_url",
            "_ALLOWED_HOST",
            "ALLOWED_HOST",
            "validate_github_url",
        ),
        "size cap on downloaded payloads (~5 MiB)": (
            "_SIZE_LIMIT",
            "SIZE_LIMIT",
            "5 * 1024 * 1024",
            "5242880",
        ),
        "--dry-run flag wired into argparse": ("--dry-run",),
        "--ref flag for SHA pinning": ("--ref",),
        "lock file written for synced prompts": (
            "flow-prompts.lock",
            "LOCK_REL",
        ),
        "license-header injection (CC BY 4.0 / Daniel Agrici)": (
            "CC BY 4.0",
            "Daniel Agrici",
        ),
        "path-traversal guard on resolved paths": (
            "is_relative_to",
            ".resolve()",
            'reject "..',
            "'..'",
            '".."',
        ),
    }

    missing = [
        label
        for label, patterns in invariants.items()
        if not _check_pattern(source, *patterns)
    ]

    assert not missing, (
        f"scripts/sync_flow.py is missing required security primitives. "
        f"Add the following before merging:\n  - " + "\n  - ".join(missing)
    )
