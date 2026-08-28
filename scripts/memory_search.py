#!/usr/bin/env python3
"""
Semantic search for Claude Code hybrid memory.
Finds relevant past sessions, topic memories, and daily logs.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import chromadb
import chromadb.errors
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()

# Configuration
DB_PATH = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "claude_memory"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
SUMMARIES_DIR = Path.home() / ".claude" / "compacted-summaries"


class MemorySearcher:
    def __init__(self):
        try:
            self.client = chromadb.PersistentClient(path=str(DB_PATH))
            ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
            self.collection = self.client.get_collection(name=COLLECTION_NAME, embedding_function=ef)
        except chromadb.errors.NotFoundError:
            console.print(f"[red]Collection '{COLLECTION_NAME}' not found.[/red]")
            console.print("[yellow]Run: python scripts/index_summaries.py[/yellow]")
            sys.exit(1)
        except Exception as e:
            console.print(f"[red]Error connecting to database: {e}[/red]")
            sys.exit(1)

    def calculate_recency_score(self, date_str: str) -> float:
        if not date_str:
            return 0.5
        try:
            session_date = datetime.strptime(date_str, "%Y-%m-%d")
            days_ago = (datetime.now() - session_date).days
            if days_ago <= 7:
                return 1.0
            elif days_ago <= 30:
                return 0.8
            elif days_ago <= 90:
                return 0.6
            elif days_ago <= 180:
                return 0.4
            else:
                return 0.2
        except Exception:
            return 0.5

    def search(
        self,
        query: str,
        n_results: int = 5,
        min_similarity: float = 0.3,
        source_filter: str | None = None,
    ) -> list[dict]:
        """Search with optional source filtering.

        `source_filter` accepts one source or several separated by commas
        ("session,daily,summary"). The multi-source form is what the narrative archive needs:
        checkouts, daily logs and compacted summaries are the corpus grep cannot reach by
        keyword, while topic files stay on the grep + frontmatter-description route that
        Claude Code is designed around.
        """
        where_filter = None
        if source_filter:
            sources = [s.strip() for s in source_filter.split(",") if s.strip()]
            if len(sources) == 1:
                where_filter = {"source_type": sources[0]}
            elif sources:
                where_filter = {"source_type": {"$in": sources}}

        # Files are indexed as ~110-token chunks (see index_summaries.py § Chunking), so a single
        # file can occupy many of the nearest neighbours. Over-fetch, then collapse to one result
        # per file keeping its best-scoring chunk — otherwise three hits could all be one document.
        results = self.collection.query(
            query_texts=[query],
            n_results=max(n_results * 20, 60),
            include=["metadatas", "documents", "distances"],
            where=where_filter,
        )

        if not results["ids"][0]:
            return []

        processed = []
        for i, doc_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i]
            similarity = 1 / (1 + distance)

            if similarity < min_similarity:
                continue

            metadata = results["metadatas"][0][i]
            document = results["documents"][0][i]

            recency = self.calculate_recency_score(metadata.get("session_date"))
            complexity_bonus = 0.1 if metadata.get("complexity") == "high" else 0

            # Source type bonus: topic files get slight boost (curated knowledge)
            source_bonus = 0.05 if metadata.get("source_type") == "topic" else 0

            hybrid_score = (
                0.65 * similarity
                + 0.20 * recency
                + complexity_bonus
                + source_bonus
            )

            # Extract preview
            lines = document.split("\n")
            preview_lines = [
                line for line in lines[:15]
                if line.strip()
                and not line.startswith("Title:")
                and not line.startswith("Description:")
                and not line.startswith("Source:")
                and not line.startswith("Type:")
                and not line.startswith("Technologies:")
            ]
            preview = "\n".join(preview_lines[:5])

            processed.append({
                "doc_id": doc_id,
                "filename": metadata.get("filename", "unknown"),
                "title": metadata.get("title", "Untitled"),
                "date": metadata.get("session_date", "unknown"),
                "source_type": metadata.get("source_type", "unknown"),
                "similarity": similarity,
                "hybrid_score": hybrid_score,
                "complexity": metadata.get("complexity", "unknown"),
                "technologies": json.loads(metadata.get("technologies", "[]")),
                "preview": preview,
                "memory_type": metadata.get("memory_type", ""),
                "description": metadata.get("description", ""),
                "parent_id": metadata.get("parent_id", ""),
                "chunk_index": metadata.get("chunk_index", 0),
            })

        processed.sort(key=lambda x: x["hybrid_score"], reverse=True)

        # Collapse chunks back to files: best chunk wins, and its text becomes the preview — which
        # is the point of chunking, since that excerpt is the passage that actually matched rather
        # than the first 15 lines of the file.
        best_per_file: dict[str, dict] = {}
        for r in processed:
            key = r.get("parent_id") or r["filename"]
            if key not in best_per_file:
                r["chunks_matched"] = 1
                best_per_file[key] = r
            else:
                best_per_file[key]["chunks_matched"] += 1

        collapsed = sorted(best_per_file.values(), key=lambda x: x["hybrid_score"], reverse=True)
        return collapsed[:n_results]

    def display_results(self, results: list[dict], query: str) -> None:
        if not results:
            console.print("[yellow]No relevant memories found.[/yellow]")
            return

        console.print(
            f"\n[cyan]Found {len(results)} result(s) for:[/cyan] '{query}'\n"
        )

        for i, r in enumerate(results, 1):
            source_icon = {"summary": "S", "topic": "T", "daily": "D"}.get(
                r["source_type"], "?"
            )
            title = f"{i}. [{source_icon}] {r['title']}"

            content = f"**Source:** {r['source_type']} | **Date:** {r['date']}\n"
            content += f"**Similarity:** {r['similarity']:.1%} | **Relevance:** {r['hybrid_score']:.1%}\n"
            if r["description"]:
                content += f"**Description:** {r['description']}\n"
            if r["technologies"]:
                content += f"**Tech:** {', '.join(r['technologies'])}\n"
            content += f"\n```\n{r['preview'][:300]}\n```"

            border = "green" if r["similarity"] > 0.5 else "yellow"
            console.print(Panel(Markdown(content), title=title, border_style=border))

    def display_compact(self, results: list[dict], query: str) -> None:
        """Compact output optimized for Claude Code context injection."""
        if not results:
            print("No relevant memories found.")
            return

        print(f"## Memory Search: '{query}' ({len(results)} results)\n")
        for i, r in enumerate(results, 1):
            source_tag = r["source_type"].upper()
            print(f"### {i}. [{source_tag}] {r['title']} ({r['date']})")
            print(f"Relevance: {r['hybrid_score']:.0%} | Similarity: {r['similarity']:.0%}")
            if r["description"]:
                print(f"Description: {r['description']}")
            print(f"\n{r['preview'][:400]}\n")


RETRIEVAL_LOG = Path(__file__).parent.parent / "logs" / "retrievals.jsonl"


def log_retrieval(query: str, source_filter: str | None, caller: str, results: list) -> None:
    """Append one line per search so the layer can be judged on evidence later.

    Until 2026-08-28 this store was indexed every six hours and never read once — there was no
    record of a single search. Without a log, "is the vector layer worth keeping?" stays a matter
    of opinion. Failures here are swallowed: logging must never break a search.
    """
    try:
        RETRIEVAL_LOG.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "query": query,
            "source": source_filter,
            "caller": caller,
            "hits": [
                {
                    "file": r.get("filename"),
                    "source": r.get("source_type"),
                    "score": round(r.get("hybrid_score", 0), 3),
                }
                for r in results[:3]
            ],
            "n": len(results),
        }
        with RETRIEVAL_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def main():
    if len(sys.argv) < 2:
        console.print("[red]Usage: python memory_search.py <query> [--source session,daily,summary|topic] [--compact] [--caller NAME][/red]")
        sys.exit(1)

    # Parse args
    args = sys.argv[1:]
    source_filter = None
    compact = False
    caller = "manual"
    query_parts = []

    i = 0
    while i < len(args):
        if args[i] == "--source" and i + 1 < len(args):
            source_filter = args[i + 1]
            i += 2
        elif args[i] == "--caller" and i + 1 < len(args):
            caller = args[i + 1]
            i += 2
        elif args[i] == "--compact":
            compact = True
            i += 1
        else:
            query_parts.append(args[i])
            i += 1

    query = " ".join(query_parts)
    if not query:
        console.print("[red]Please provide a search query.[/red]")
        sys.exit(1)

    searcher = MemorySearcher()
    results = searcher.search(query, n_results=3, source_filter=source_filter)
    log_retrieval(query, source_filter, caller, results)

    if compact:
        searcher.display_compact(results, query)
    else:
        searcher.display_results(results, query)


if __name__ == "__main__":
    main()
