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
        """Search with optional source filtering (summary, topic, daily)."""
        where_filter = None
        if source_filter:
            where_filter = {"source_type": source_filter}

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results * 2,
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
            })

        processed.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return processed[:n_results]

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


def main():
    if len(sys.argv) < 2:
        console.print("[red]Usage: python memory_search.py <query> [--source summary|topic|daily] [--compact][/red]")
        sys.exit(1)

    # Parse args
    args = sys.argv[1:]
    source_filter = None
    compact = False
    query_parts = []

    i = 0
    while i < len(args):
        if args[i] == "--source" and i + 1 < len(args):
            source_filter = args[i + 1]
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

    if compact:
        searcher.display_compact(results, query)
    else:
        searcher.display_results(results, query)


if __name__ == "__main__":
    main()
