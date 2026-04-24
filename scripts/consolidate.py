#!/usr/bin/env python3
"""
Memory Consolidator — weekly cleanup of Auto Memory files.

Analyzes MEMORY.md and topic files for:
  - Duplicate/overlapping entries
  - Stale references (files/features that no longer exist)
  - Entries that should be in topic files (not MEMORY.md index)
  - MEMORY.md line count (target: under 200 lines)

Outputs a report with suggested changes. Does NOT auto-apply.
Can optionally use Claude API for intelligent analysis.
"""

import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Memory paths
MEMORY_DIR = (
    Path.home()
    / ".claude"
    / "projects"
    / "-Users-Syncopation-Documents-Git-Local"
    / "memory"
)
MEMORY_INDEX = MEMORY_DIR / "MEMORY.md"
REPO_ROOT = Path.home() / "Documents" / "Git-Local"

# Thresholds
MAX_INDEX_LINES = 200
WARNING_LINES = 180


def count_lines(filepath: Path) -> int:
    if filepath.exists():
        return len(filepath.read_text().splitlines())
    return 0


def find_file_references(content: str) -> list[str]:
    """Extract file path references from content."""
    patterns = [
        r"`([^`]*(?:\.(?:ts|js|py|md|json|yaml|sh))[^`]*)`",
        r"(?:Path|path|File|file):\s*`?([^\s`]+\.(?:ts|js|py|md|json|yaml|sh))`?",
    ]
    refs = set()
    for pattern in patterns:
        refs.update(re.findall(pattern, content))
    return list(refs)


def check_stale_references(content: str) -> list[dict]:
    """Find file references that no longer exist."""
    stale = []
    refs = find_file_references(content)
    for ref in refs:
        # Skip relative paths without clear root, URLs, etc.
        if ref.startswith("http") or ref.startswith("@") or "/" not in ref:
            continue

        # Try common roots
        candidates = [
            REPO_ROOT / ref,
            REPO_ROOT / ref.lstrip("/"),
            Path.home() / ref.lstrip("/"),
        ]
        exists = any(c.exists() for c in candidates)
        if not exists:
            stale.append({"ref": ref, "status": "not_found"})

    return stale


def find_duplicate_topics(index_content: str, topic_files: list[Path]) -> list[dict]:
    """Find entries in MEMORY.md that duplicate topic file content."""
    dupes = []
    index_lines = index_content.splitlines()

    for topic_file in topic_files:
        topic_name = topic_file.stem.replace("-", " ").replace("_", " ").lower()
        # Check if topic is mentioned multiple times in index
        mentions = [
            (i, line) for i, line in enumerate(index_lines, 1)
            if topic_name in line.lower() and not line.startswith("#")
        ]
        if len(mentions) > 2:
            dupes.append({
                "topic": topic_file.name,
                "mentions": len(mentions),
                "lines": [m[0] for m in mentions],
            })

    return dupes


def find_long_entries(index_content: str) -> list[dict]:
    """Find entries in MEMORY.md that are too detailed for an index."""
    long = []
    lines = index_content.splitlines()

    # Find sections that are too verbose (>5 lines of detail)
    current_section = None
    section_lines = 0

    for i, line in enumerate(lines, 1):
        if line.startswith("## "):
            if current_section and section_lines > 15:
                long.append({
                    "section": current_section,
                    "lines": section_lines,
                    "suggestion": "Move details to topic file",
                })
            current_section = line.strip("# ").strip()
            section_lines = 0
        elif current_section:
            section_lines += 1

    # Check last section
    if current_section and section_lines > 15:
        long.append({
            "section": current_section,
            "lines": section_lines,
            "suggestion": "Move details to topic file",
        })

    return long


def analyze_memory() -> dict:
    """Run full analysis of memory system."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "index_lines": 0,
        "topic_files": 0,
        "issues": [],
        "suggestions": [],
    }

    if not MEMORY_INDEX.exists():
        report["issues"].append("MEMORY.md not found")
        return report

    index_content = MEMORY_INDEX.read_text()
    report["index_lines"] = count_lines(MEMORY_INDEX)

    # List topic files
    topic_files = [
        f for f in MEMORY_DIR.glob("*.md")
        if f.name not in ("MEMORY.md", "FACTS.md")
    ]
    report["topic_files"] = len(topic_files)

    # Check 1: Line count
    if report["index_lines"] > MAX_INDEX_LINES:
        report["issues"].append(
            f"MEMORY.md is {report['index_lines']} lines (limit: {MAX_INDEX_LINES}). "
            f"Need to move {report['index_lines'] - MAX_INDEX_LINES} lines to topic files."
        )

    # Check 2: Stale file references
    stale = check_stale_references(index_content)
    if stale:
        report["issues"].append(
            f"Found {len(stale)} stale file references: "
            + ", ".join(s["ref"] for s in stale[:5])
        )
        report["stale_refs"] = stale

    # Check 3: Duplicate topics
    dupes = find_duplicate_topics(index_content, topic_files)
    if dupes:
        for d in dupes:
            report["suggestions"].append(
                f"'{d['topic']}' is mentioned {d['mentions']} times in index — "
                f"consolidate into topic file"
            )

    # Check 4: Long sections
    long = find_long_entries(index_content)
    if long:
        for entry in long:
            report["suggestions"].append(
                f"Section '{entry['section']}' has {entry['lines']} lines — "
                f"{entry['suggestion']}"
            )
        report["long_sections"] = long

    # Check 5: Topic files without index pointers
    for tf in topic_files:
        tf_stem = tf.stem.replace("-", " ").replace("_", " ")
        if tf_stem.lower() not in index_content.lower() and tf.name not in index_content:
            report["suggestions"].append(
                f"Topic file '{tf.name}' has no pointer in MEMORY.md index"
            )

    # Check 6: Relative dates (should be absolute)
    relative_dates = re.findall(
        r"(?:yesterday|today|last week|this week|next week|tomorrow)",
        index_content,
        re.IGNORECASE,
    )
    if relative_dates:
        report["issues"].append(
            f"Found {len(relative_dates)} relative dates in MEMORY.md — "
            "should be converted to absolute dates (e.g., 2026-03-25)"
        )

    return report


def print_report(report: dict) -> None:
    """Print human-readable consolidation report."""
    print("=" * 60)
    print("  MEMORY CONSOLIDATION REPORT")
    print(f"  {report['timestamp']}")
    print("=" * 60)
    print()

    # Status
    status = "OK" if report["index_lines"] <= MAX_INDEX_LINES else "OVER LIMIT"
    print(f"MEMORY.md: {report['index_lines']}/{MAX_INDEX_LINES} lines [{status}]")
    print(f"Topic files: {report['topic_files']}")
    print()

    # Issues
    if report["issues"]:
        print("ISSUES:")
        for i, issue in enumerate(report["issues"], 1):
            print(f"  {i}. {issue}")
        print()

    # Suggestions
    if report["suggestions"]:
        print("SUGGESTIONS:")
        for i, sugg in enumerate(report["suggestions"], 1):
            print(f"  {i}. {sugg}")
        print()

    # Summary
    total = len(report["issues"]) + len(report["suggestions"])
    if total == 0:
        print("Memory system is healthy. No changes needed.")
    else:
        print(f"Total: {len(report['issues'])} issues, {len(report['suggestions'])} suggestions")
        print()
        print("To apply changes:")
        print("  1. Review suggestions above")
        print("  2. Run: claude '/memory-consolidate apply' (when skill is ready)")
        print("  3. Or manually edit MEMORY.md and topic files")


def main():
    report = analyze_memory()
    print_report(report)

    # Return non-zero if issues found (useful for CI)
    if report["issues"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
