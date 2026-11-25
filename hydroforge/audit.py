# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2026 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Extract and validate structured upstream-model bug-fix annotations.

Downstream HydroForge models often uncover genuine physical, dimensional,
conservation, or boundary-condition bugs in upstream reference codes. To make
every intentional divergence auditable across Python, CUDA, Metal, and Fortran
sources, annotate each site with a compact ``HF-UPSTREAM-FIX`` comment block::

    # HF-UPSTREAM-FIX[CMF-01]: Short summary of the physical correction
    #   upstream: CaMa-Flood v4.30 cmf_calc_outflw_mod.F90:95 - double-subtracts channel area
    #   fix: Use flood_storage / river_length directly as floodplain cross-sectional area
    #   mode: fixed        # optional: fixed (default) | gated | parity
    #   option: ...        # required when mode is gated
    #   impact: hydraulics # optional: short impact category

Comment prefixes ``#``, ``//``, and ``!`` are all supported. Python files are
tokenized so examples inside docstrings or string literals are ignored.
Multi-backend sites sharing the same ``[ID]`` (e.g. CUDA, Triton, and Metal)
are automatically grouped and cross-checked for mode/option consistency even
when output is filtered by ``--mode``.

Run from the command line::

    python -m hydroforge.audit path/to/model [more/paths ...] --format md --check
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import tokenize
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

TAG = "HF-UPSTREAM-FIX"
VALID_MODES = frozenset({"fixed", "gated", "parity"})
DEFAULT_EXTENSIONS = frozenset(
    {
        ".py",
        ".cu",
        ".cuh",
        ".metal",
        ".c",
        ".cc",
        ".cpp",
        ".h",
        ".hpp",
        ".f",
        ".F",
        ".f90",
        ".F90",
    }
)
DEFAULT_EXCLUDE_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        "build",
        "dist",
        "node_modules",
        ".eggs",
    }
)

_HEADER_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<prefix>#|//|!)[ \t]*HF-UPSTREAM-FIX"
    r"\[(?P<id>[A-Za-z0-9._-]+)\]:[ \t]*(?P<summary>.+?)[ \t]*$"
)
_LOOSE_TAG_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<prefix>#|//|!)[ \t]*HF-UPSTREAM-FIX\b"
)
_FIELD_RE = re.compile(
    r"^(?P<pad>[ \t]*)(?P<key>upstream|fix|mode|option|impact):[ \t]*(?P<value>.*?)[ \t]*$"
)


@dataclass(frozen=True)
class UpstreamFix:
    """One ``HF-UPSTREAM-FIX`` annotation extracted from a source file."""

    id: str
    summary: str
    upstream: str
    fix: str
    mode: str
    option: str
    impact: str
    file: str
    line: int
    repo: str

    @property
    def location(self) -> str:
        return f"{self.file}:{self.line}"


@dataclass(frozen=True)
class GroupedFix:
    """Logical fix grouped by ``id`` across backends or files."""

    id: str
    summary: str
    upstream: str
    fix: str
    mode: str
    option: str
    impact: str
    repos: tuple[str, ...]
    locations: tuple[str, ...]


@dataclass(frozen=True)
class ValidationIssue:
    """A malformed, missing-path, or inconsistent ``HF-UPSTREAM-FIX`` check finding."""

    file: str
    line: int
    message: str


@dataclass(frozen=True)
class ScanResult:
    """Aggregated extraction and validation result."""

    entries: tuple[UpstreamFix, ...]
    grouped: tuple[GroupedFix, ...]
    issues: tuple[ValidationIssue, ...]
    files_scanned: int


def _iter_source_files(
    paths: Sequence[Path],
    extensions: frozenset[str],
    exclude_dirs: frozenset[str],
    issues: list[ValidationIssue],
) -> Iterable[tuple[str, Path, Path]]:
    """Yield ``(repo_label, root_dir, file_path)`` in deterministic sorted order.

    Records a :class:`ValidationIssue` for any missing or non-file/non-directory
    input path instead of silently skipping it.
    """
    seen: set[Path] = set()
    for raw in paths:
        if not raw.exists():
            issues.append(
                ValidationIssue(
                    file=str(raw),
                    line=0,
                    message=f"input path does not exist: {raw}",
                )
            )
            continue
        target = raw.resolve()
        if target.is_file():
            if target not in seen:
                seen.add(target)
                yield target.parent.name, target.parent, target
            continue
        if not target.is_dir():
            issues.append(
                ValidationIssue(
                    file=str(raw),
                    line=0,
                    message=f"input path is neither a file nor a directory: {raw}",
                )
            )
            continue
        repo_label = target.name
        stack = [target]
        found: list[Path] = []
        while stack:
            current = stack.pop()
            for child in sorted(current.iterdir()):
                if child.is_dir():
                    if child.name in exclude_dirs or child.name.endswith(".egg-info"):
                        continue
                    stack.append(child)
                elif child.is_file() and child.suffix in extensions:
                    found.append(child)
        for path in sorted(found):
            if path not in seen:
                seen.add(path)
                yield repo_label, target, path


def _python_comment_line_numbers(text: str) -> set[int] | None:
    """Return 1-based line numbers of standalone Python comment lines."""
    comment_lines: set[int] = set()
    code_lines: set[int] = set()
    ignored = {
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.ENDMARKER,
        tokenize.ENCODING,
    }
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.COMMENT:
                comment_lines.add(tok.start[0])
            elif tok.type not in ignored:
                for ln in range(tok.start[0], tok.end[0] + 1):
                    code_lines.add(ln)
    except (tokenize.TokenError, IndentationError):
        return None
    return comment_lines - code_lines


def _indent_width(text: str) -> int:
    return len(text.expandtabs(4)) - len(text.expandtabs(4).lstrip(" "))


def parse_file_annotations(
    text: str,
    *,
    rel_file: str,
    repo: str = "",
) -> tuple[list[UpstreamFix], list[ValidationIssue]]:
    """Parse all ``HF-UPSTREAM-FIX`` blocks from one source text."""
    entries: list[UpstreamFix] = []
    issues: list[ValidationIssue] = []
    lines = text.splitlines()
    py_comments = _python_comment_line_numbers(text) if rel_file.endswith(".py") else None
    n = len(lines)
    i = 0

    while i < n:
        line_no = i + 1
        if py_comments is not None and line_no not in py_comments:
            i += 1
            continue

        line = lines[i]
        m = _HEADER_RE.match(line)
        if not m:
            if _LOOSE_TAG_RE.match(line):
                issues.append(
                    ValidationIssue(
                        file=rel_file,
                        line=line_no,
                        message=(
                            "Malformed header; expected "
                            "'HF-UPSTREAM-FIX[<ID>]: <summary>'"
                        ),
                    )
                )
            i += 1
            continue

        start_line = line_no
        indent = m.group("indent")
        prefix = m.group("prefix")
        fix_id = m.group("id")
        summary = m.group("summary").strip()
        leader = f"{indent}{prefix}"

        fields: dict[str, list[str]] = {}
        current_key: str | None = None
        current_key_indent = 0
        i += 1

        while i < n:
            if py_comments is not None and (i + 1) not in py_comments:
                break
            nxt = lines[i]
            if not nxt.startswith(leader):
                break
            if _LOOSE_TAG_RE.match(nxt):
                break
            body = nxt[len(leader) :]
            if not body.strip():
                break
            fm = _FIELD_RE.match(body)
            if fm:
                current_key = fm.group("key")
                current_key_indent = _indent_width(fm.group("pad"))
                val = fm.group("value").strip()
                fields.setdefault(current_key, [])
                if val:
                    fields[current_key].append(val)
                i += 1
                continue
            if current_key is not None and _indent_width(body) > current_key_indent:
                fields[current_key].append(body.strip())
                i += 1
                continue
            break

        upstream = " ".join(fields.get("upstream", ())).strip()
        fix = " ".join(fields.get("fix", ())).strip()
        mode = " ".join(fields.get("mode", ())).strip().lower() or "fixed"
        option = " ".join(fields.get("option", ())).strip()
        impact = " ".join(fields.get("impact", ())).strip()

        if not upstream:
            issues.append(
                ValidationIssue(
                    file=rel_file,
                    line=start_line,
                    message=f"[{fix_id}] missing required 'upstream:' field",
                )
            )
        if not fix:
            issues.append(
                ValidationIssue(
                    file=rel_file,
                    line=start_line,
                    message=f"[{fix_id}] missing required 'fix:' field",
                )
            )
        if mode not in VALID_MODES:
            issues.append(
                ValidationIssue(
                    file=rel_file,
                    line=start_line,
                    message=(
                        f"[{fix_id}] invalid mode {mode!r}; "
                        f"expected one of {sorted(VALID_MODES)}"
                    ),
                )
            )
        if mode == "gated" and not option:
            issues.append(
                ValidationIssue(
                    file=rel_file,
                    line=start_line,
                    message=f"[{fix_id}] mode='gated' requires an 'option:' field",
                )
            )

        entries.append(
            UpstreamFix(
                id=fix_id,
                summary=summary,
                upstream=upstream,
                fix=fix,
                mode=mode,
                option=option,
                impact=impact,
                file=rel_file,
                line=start_line,
                repo=repo,
            )
        )

    return entries, issues


def group_fixes(
    entries: Sequence[UpstreamFix],
) -> tuple[tuple[GroupedFix, ...], list[ValidationIssue]]:
    """Group annotations by ``id`` and check cross-site consistency."""
    by_id: dict[str, list[UpstreamFix]] = defaultdict(list)
    for entry in entries:
        by_id[entry.id].append(entry)

    grouped: list[GroupedFix] = []
    issues: list[ValidationIssue] = []

    for fix_id in sorted(by_id):
        group = by_id[fix_id]
        modes = {e.mode for e in group}
        if len(modes) > 1:
            issues.append(
                ValidationIssue(
                    file=group[0].file,
                    line=group[0].line,
                    message=(
                        f"[{fix_id}] conflicting modes across sites: {sorted(modes)}"
                    ),
                )
            )
        options = {e.option for e in group if e.option}
        if len(options) > 1:
            issues.append(
                ValidationIssue(
                    file=group[0].file,
                    line=group[0].line,
                    message=(
                        f"[{fix_id}] conflicting options across sites: {sorted(options)}"
                    ),
                )
            )
        head = group[0]
        repos = tuple(dict.fromkeys(e.repo for e in group if e.repo))
        locations = tuple(e.location for e in group)
        grouped.append(
            GroupedFix(
                id=fix_id,
                summary=head.summary,
                upstream=head.upstream,
                fix=head.fix,
                mode=head.mode,
                option=head.option or next((e.option for e in group if e.option), ""),
                impact=head.impact or next((e.impact for e in group if e.impact), ""),
                repos=repos,
                locations=locations,
            )
        )

    return tuple(grouped), issues


def scan_upstream_fixes(
    paths: Sequence[str | Path],
    *,
    extensions: frozenset[str] = DEFAULT_EXTENSIONS,
    exclude_dirs: frozenset[str] = DEFAULT_EXCLUDE_DIRS,
    mode_filter: str | None = None,
    id_prefix: str | None = None,
) -> ScanResult:
    """Scan one or more files/directories for ``HF-UPSTREAM-FIX`` blocks.

    Cross-site consistency (such as conflicting ``mode`` or ``option`` values
    for the same fix ID across backends) is always validated BEFORE applying
    ``mode_filter`` so filtering by mode cannot mask cross-backend drift.
    """
    resolved_paths = [Path(p) for p in paths]
    all_entries: list[UpstreamFix] = []
    all_issues: list[ValidationIssue] = []
    files_scanned = 0

    for repo_label, root_dir, file_path in _iter_source_files(
        resolved_paths, extensions, exclude_dirs, all_issues
    ):
        files_scanned += 1
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            all_issues.append(
                ValidationIssue(
                    file=str(file_path), line=0, message=f"read error: {exc}"
                )
            )
            continue
        if TAG not in text:
            continue
        rel = file_path.relative_to(root_dir).as_posix()
        rel_display = f"{repo_label}/{rel}" if len(resolved_paths) > 1 else rel
        file_entries, file_issues = parse_file_annotations(
            text, rel_file=rel_display, repo=repo_label
        )
        all_entries.extend(file_entries)
        all_issues.extend(file_issues)

    # Apply id_prefix before grouping, but ALWAYS validate cross-site consistency
    # before mode_filter so --mode cannot hide cross-backend mode conflicts.
    id_scoped = (
        [e for e in all_entries if e.id.startswith(id_prefix)]
        if id_prefix
        else all_entries
    )
    full_grouped, group_issues = group_fixes(id_scoped)
    all_issues.extend(group_issues)

    if mode_filter:
        filtered_entries = [e for e in id_scoped if e.mode == mode_filter]
        grouped, _ = group_fixes(filtered_entries)
    else:
        filtered_entries = id_scoped
        grouped = full_grouped

    return ScanResult(
        entries=tuple(filtered_entries),
        grouped=grouped,
        issues=tuple(all_issues),
        files_scanned=files_scanned,
    )


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render_markdown(
    result: ScanResult, *, title: str = "Upstream Model Bug Fixes"
) -> str:
    """Render a compact Markdown report grouped by fix ID."""
    lines = [
        f"# {title}",
        "",
        (
            f"Scanned **{result.files_scanned}** files: "
            f"**{len(result.grouped)}** unique fixes across "
            f"**{len(result.entries)}** annotated sites."
        ),
        "",
    ]
    if not result.grouped:
        lines.append("_No `HF-UPSTREAM-FIX` annotations found._")
    else:
        lines.extend(
            [
                "| ID | Mode | Summary | Upstream Bug | Fix | Locations |",
                "|---|---|---|---|---|---|",
            ]
        )
        for g in result.grouped:
            mode_cell = f"`{g.mode}`" + (f" (`{g.option}`)" if g.option else "")
            locs = "<br>".join(f"`{loc}`" for loc in g.locations)
            lines.append(
                f"| `{_md_escape(g.id)}` | {mode_cell} | {_md_escape(g.summary)} | "
                f"{_md_escape(g.upstream)} | {_md_escape(g.fix)} | {locs} |"
            )

    if result.issues:
        lines.extend(["", "## Validation Issues", ""])
        for issue in result.issues:
            loc = f"{issue.file}:{issue.line}" if issue.line else issue.file
            lines.append(f"- `{loc}`: {issue.message}")

    return "\n".join(lines) + "\n"


def render_text(result: ScanResult) -> str:
    """Render a concise plain-text terminal summary."""
    lines = [
        f"HF-UPSTREAM-FIX audit: {len(result.grouped)} unique fixes "
        f"({len(result.entries)} sites) across {result.files_scanned} files"
    ]
    for g in result.grouped:
        extra = f" option={g.option}" if g.option else ""
        impact = f" impact={g.impact}" if g.impact else ""
        lines.append(f"\n[{g.id}] ({g.mode}{extra}{impact}) {g.summary}")
        lines.append(f"  upstream : {g.upstream}")
        lines.append(f"  fix      : {g.fix}")
        lines.append(f"  sites    : {', '.join(g.locations)}")
    if result.issues:
        lines.append(f"\nValidation issues ({len(result.issues)}):")
        for issue in result.issues:
            loc = f"{issue.file}:{issue.line}" if issue.line else issue.file
            lines.append(f"  ! {loc}: {issue.message}")
    return "\n".join(lines) + "\n"


def render_json(result: ScanResult) -> str:
    """Render machine-readable JSON output."""
    payload = {
        "files_scanned": result.files_scanned,
        "unique_fixes": len(result.grouped),
        "total_sites": len(result.entries),
        "grouped": [asdict(g) for g in result.grouped],
        "entries": [asdict(e) for e in result.entries],
        "issues": [asdict(i) for i in result.issues],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m hydroforge.audit",
        description="Extract and validate HF-UPSTREAM-FIX annotations across model repos.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Model repository directories or source files to scan.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "md", "json"),
        default="text",
        help="Output format (default: text).",
    )
    parser.add_argument(
        "--mode",
        choices=sorted(VALID_MODES),
        default=None,
        help="Filter fixes by mode.",
    )
    parser.add_argument(
        "--id",
        dest="id_prefix",
        default=None,
        help="Filter fixes whose ID starts with this prefix (e.g. CMF-).",
    )
    parser.add_argument(
        "--title",
        default="Upstream Model Bug Fixes",
        help="Report heading for Markdown output.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write rendered output to this file instead of stdout.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit with code 1 if any annotation validation issue is detected.",
    )
    args = parser.parse_args(argv)

    missing_paths = [p for p in args.paths if not p.exists()]
    result = scan_upstream_fixes(
        args.paths,
        mode_filter=args.mode,
        id_prefix=args.id_prefix,
    )
    if args.format == "md":
        rendered = render_markdown(result, title=args.title)
    elif args.format == "json":
        rendered = render_json(result)
    else:
        rendered = render_text(result)

    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)

    if missing_paths or (args.check and result.issues):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
