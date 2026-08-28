#!/usr/bin/env python3
"""
Index all memory sources into ChromaDB for semantic search.

Sources:
  1. Compacted summaries: ~/.claude/compacted-summaries/*.md
  2. Auto Memory topic files: ~/.claude/projects/.../memory/*.md
  3. Daily logs: 00-core/.context/memory/daily/*.md
  4. Session checkpoints: 00-core/.context/session/checkout-*.md

Supports incremental indexing (only new/changed files).
"""

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rich.console import Console
from rich.progress import track
from rich.table import Table

console = Console()

# Configuration
DB_PATH = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "claude_memory"  # New name to avoid conflicts with old collection
STATE_FILE = Path(__file__).parent.parent / "index_state.json"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Source directories
SUMMARIES_DIR = Path.home() / ".claude" / "compacted-summaries"
TOPIC_FILES_DIR = (
    Path.home()
    / ".claude"
    / "projects"
    / "-Users-Syncopation-Documents-Git-Local"
    / "memory"
)
DAILY_LOGS_DIR = (
    Path(__file__).parent.parent.parent.parent
    / "00-core"
    / ".context"
    / "memory"
    / "daily"
)

# Resolve DAILY_LOGS_DIR relative to repo root if above path doesn't exist
if not DAILY_LOGS_DIR.exists():
    DAILY_LOGS_DIR = (
        Path.home()
        / "Documents"
        / "Git-Local"
        / "00-core"
        / ".context"
        / "memory"
        / "daily"
    )

SESSION_DIR = (
    Path(__file__).parent.parent.parent.parent
    / "00-core"
    / ".context"
    / "session"
)
if not SESSION_DIR.exists():
    SESSION_DIR = (
        Path.home()
        / "Documents"
        / "Git-Local"
        / "00-core"
        / ".context"
        / "session"
    )

SOURCES = {
    "summary": {"dir": SUMMARIES_DIR, "prefix": "summary:"},
    "topic": {"dir": TOPIC_FILES_DIR, "prefix": "topic:"},
    "daily": {"dir": DAILY_LOGS_DIR, "prefix": "daily:"},
    "session": {"dir": SESSION_DIR, "prefix": "session:", "glob": "checkout-*.md"},
}

# Tech keywords for metadata extraction
TECH_KEYWORDS = [
    "vue", "typescript", "python", "react", "node", "prisma",
    "vitest", "playwright", "tailwind", "sqlite", "chromadb",
    "grammy", "telegram", "nextjs", "supabase", "docker",
    "github", "claude", "anthropic", "mcp", "odoo", "kommo",
]


def load_index_state() -> dict:
    """Load previous indexing state (file mtimes)."""
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_index_state(state: dict) -> None:
    """Save indexing state for incremental updates."""
    STATE_FILE.write_text(json.dumps(state, indent=2))


def extract_metadata(content: str, filename: str, source_type: str) -> dict:
    """Extract metadata from file content and filename."""
    metadata = {
        "filename": filename,
        "source_type": source_type,
        "indexed_at": datetime.now().isoformat(),
    }

    # Extract date from filename or content
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    if date_match:
        metadata["session_date"] = date_match.group(1)
    else:
        # Try content for date references
        date_in_content = re.search(r"(\d{4}-\d{2}-\d{2})", content[:500])
        if date_in_content:
            metadata["session_date"] = date_in_content.group(1)

    # Extract title
    title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if title_match:
        metadata["title"] = title_match.group(1).strip()[:200]
    else:
        # Use filename as title for topic files
        metadata["title"] = filename.replace(".md", "").replace("-", " ").replace("_", " ")

    # Extract frontmatter fields (for Auto Memory topic files)
    name_match = re.search(r"^name:\s*(.+)$", content, re.MULTILINE)
    if name_match:
        metadata["title"] = name_match.group(1).strip()

    desc_match = re.search(r"^description:\s*(.+)$", content, re.MULTILINE)
    if desc_match:
        metadata["description"] = desc_match.group(1).strip()[:300]

    type_match = re.search(r"^type:\s*(.+)$", content, re.MULTILINE)
    if type_match:
        metadata["memory_type"] = type_match.group(1).strip()

    # Complexity scoring
    content_length = len(content)
    file_refs = len(re.findall(r"\.(?:ts|js|py|md|json|yaml)", content))
    if content_length > 3000 or file_refs > 10:
        metadata["complexity"] = "high"
    elif content_length > 1500 or file_refs > 5:
        metadata["complexity"] = "medium"
    else:
        metadata["complexity"] = "low"

    # Technology detection
    found_techs = [t for t in TECH_KEYWORDS if t in content.lower()]
    if found_techs:
        metadata["technologies"] = json.dumps(found_techs)

    return metadata


def create_document_text(content: str, metadata: dict) -> str:
    """Create searchable document text combining content and metadata."""
    parts = []
    if "title" in metadata:
        parts.append(f"Title: {metadata['title']}")
    if "description" in metadata:
        parts.append(f"Description: {metadata['description']}")
    if "memory_type" in metadata:
        parts.append(f"Type: {metadata['memory_type']}")
    if "technologies" in metadata:
        techs = json.loads(metadata["technologies"])
        parts.append(f"Technologies: {', '.join(techs)}")
    parts.append(f"Source: {metadata['source_type']}")
    parts.append("")
    parts.append(content)
    return "\n".join(parts)


# --- Chunking -------------------------------------------------------------------------------
#
# The embedding model's window is 128 word-piece tokens. Until 2026-08-28 this indexer stored ONE
# embedding per whole file, so everything past the first ~128 tokens was invisible to search: a
# 31 460-token daily log was represented by 0.41 % of itself, and the dated archive files created
# that day embedded only their identical split headers (pairwise cosine up to 0.997). The layer
# looked configured and was effectively blind — a search for a fact quoted verbatim in an archive
# returned it 7th out of 10.
#
# Calibrated by measurement, not by arithmetic. A first attempt used 330 content characters plus
# a 60-character overlap on the assumption of 3.08 chars/token; measuring the real chunks gave a
# MEDIAN of 129 tokens and a maximum of 181 — 122 of 228 chunks were still being truncated by the
# very window this change exists to respect. Markdown with paths, code spans and issue numbers
# runs closer to 2.5 chars/token than prose does. Budget below is content + overlap together, and
# the acceptance test asserts max <= 128 on real files.
CHARS_PER_TOKEN = 2.5
CHUNK_CHARS = 230
CHUNK_OVERLAP_CHARS = 45


def chunk_content(content: str) -> list[str]:
    """Split file content into overlapping, window-sized chunks on line boundaries.

    Line boundaries are kept because these files are Markdown logs: a chunk that starts mid
    sentence embeds worse than one that starts at a heading or a bullet. A single line longer
    than the budget is hard-split — a 900-character paragraph is common in these notes.
    """
    chunks: list[str] = []
    buf: list[str] = []
    size = 0

    def flush() -> None:
        nonlocal buf, size
        if buf:
            chunks.append("\n".join(buf).strip())
            buf, size = [], 0

    for line in content.splitlines():
        if len(line) > CHUNK_CHARS:
            flush()
            for i in range(0, len(line), CHUNK_CHARS - CHUNK_OVERLAP_CHARS):
                piece = line[i : i + CHUNK_CHARS].strip()
                if piece:
                    chunks.append(piece)
            continue
        if size + len(line) + 1 > CHUNK_CHARS:
            flush()
        buf.append(line)
        size += len(line) + 1
    flush()

    chunks = [c for c in chunks if c]
    if not chunks:
        return []

    # Carry a tail of the previous chunk into the next one so a fact split across the boundary is
    # still retrievable from at least one side.
    overlapped = [chunks[0]]
    for prev, cur in zip(chunks, chunks[1:]):
        tail = prev[-CHUNK_OVERLAP_CHARS:]
        overlapped.append(f"{tail}\n{cur}" if tail.strip() else cur)
    return overlapped


def index_all(incremental: bool = True) -> None:
    """Index all memory sources."""
    console.print("[cyan]Initializing semantic memory system...[/cyan]")
    console.print(f"[cyan]Embedding model: {EMBEDDING_MODEL}[/cyan]")

    client = chromadb.PersistentClient(path=str(DB_PATH))
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

    # Get or create collection with multilingual embeddings
    try:
        collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)
        console.print(f"[green]Found collection: {COLLECTION_NAME} ({collection.count()} docs)[/green]")
    except Exception:
        collection = client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
            metadata={"description": "Claude Code hybrid memory", "model": EMBEDDING_MODEL},
        )
        console.print(f"[yellow]Created new collection: {COLLECTION_NAME}[/yellow]")

    # Load previous state for incremental indexing
    prev_state = load_index_state() if incremental else {}
    new_state = {}
    indexed_count = 0
    skipped_count = 0
    chunk_count = 0
    results = []

    for source_type, config in SOURCES.items():
        source_dir = config["dir"]
        prefix = config["prefix"]
        glob_pattern = config.get("glob", "*.md")

        if not source_dir.exists():
            console.print(f"[yellow]Source dir not found: {source_dir}[/yellow]")
            continue

        files = list(source_dir.glob(glob_pattern))
        console.print(f"\n[cyan]{source_type}: {len(files)} files in {source_dir}[/cyan]")

        for filepath in track(files, description=f"Indexing {source_type}..."):
            # Skip MEMORY.md index file (it's just pointers)
            if filepath.name == "MEMORY.md":
                continue

            doc_id = f"{prefix}{filepath.name}"
            mtime = os.path.getmtime(filepath)
            new_state[doc_id] = mtime

            # Skip if unchanged (incremental mode)
            if incremental and doc_id in prev_state:
                if prev_state[doc_id] == mtime:
                    skipped_count += 1
                    continue

            try:
                content = filepath.read_text(encoding="utf-8")
                if not content.strip():
                    continue

                metadata = extract_metadata(content, filepath.name, source_type)

                # Chunk count varies as a file is edited, so stale chunks from the previous run
                # must go before the new ones land — an upsert alone would leave orphans that
                # still answer queries with deleted text.
                try:
                    collection.delete(where={"parent_id": doc_id})
                except Exception:
                    pass

                pieces = chunk_content(content)
                # Chunk 0 is the file's identity card (title, description, type, source) — the
                # same text the old one-embedding-per-file scheme stored. Keeping it means a
                # query phrased like a file's description still finds the file.
                header = create_document_text("", metadata).strip()
                docs = [header] + pieces
                ids = [f"{doc_id}#{i}" for i in range(len(docs))]
                metas = [
                    {**metadata, "parent_id": doc_id, "chunk_index": i, "n_chunks": len(docs)}
                    for i in range(len(docs))
                ]

                for start in range(0, len(docs), 256):
                    collection.upsert(
                        ids=ids[start : start + 256],
                        documents=docs[start : start + 256],
                        metadatas=metas[start : start + 256],
                    )

                chunk_count += len(docs)
                indexed_count += 1
                results.append({
                    "source": source_type,
                    "filename": filepath.name,
                    "title": metadata.get("title", "N/A")[:50],
                    "complexity": metadata.get("complexity", "N/A"),
                })
            except Exception as e:
                console.print(f"[red]Error indexing {filepath.name}: {e}[/red]")

    # Save new state
    save_index_state(new_state)

    # Summary
    console.print(f"\n[green]Indexed: {indexed_count} files -> {chunk_count} chunks | Skipped (unchanged): {skipped_count} | Total in DB: {collection.count()}[/green]")

    if results:
        table = Table(title="Newly Indexed Documents")
        table.add_column("Source", style="cyan")
        table.add_column("File", style="green")
        table.add_column("Title", style="yellow")
        table.add_column("Complexity", style="magenta")

        for r in results[:20]:  # Show max 20 rows
            table.add_row(r["source"], r["filename"], r["title"], r["complexity"])

        if len(results) > 20:
            table.add_row("...", f"+ {len(results) - 20} more", "", "")

        console.print(table)


if __name__ == "__main__":
    full_rebuild = "--full" in sys.argv
    if full_rebuild:
        console.print("[yellow]Full rebuild mode (ignoring previous state)[/yellow]")
    index_all(incremental=not full_rebuild)
