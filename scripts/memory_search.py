#!/usr/bin/env python3
"""
Semantic search for Claude Code hybrid memory.
Finds relevant past sessions, topic memories, and daily logs.
"""

import json
import re
import time
from collections import Counter, defaultdict
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

# Lexical pass (added 2026-09-05). Embeddings alone are weak on proper nouns and rare terms: six real
# questions found the expected transcript at rank 4-7 or not at all while an exact substring lookup
# found it instantly ("Забродин" lost to a DistroKid dump). Cerebras's knowledge base solved the same
# gap with full-text + embeddings fused by reciprocal rank fusion; this is that idea at our scale.
TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё\-]{2,}")
CYRILLIC_RE = re.compile(r"^[А-Яа-яЁё\-]+$")
LEXICAL_MAX_TOKENS = 8       # longest tokens win; more than this costs seconds for no gain
LEXICAL_CHUNK_LIMIT = 3000   # per-token fetch cap; commonness is judged by distinct files, not chunks
LEXICAL_MAX_FILES = 80       # IDF cap: a token in more files than this carries no signal; below it, weight = 1/files
# Rank the source that *is* the knowledge above the logs that merely mention it: a meeting page beats
# the daily log that says the meeting happened. Same idea as Cerebras weighting Slack below the wiki.
LEXICAL_SOURCE_WEIGHT = {"wiki": 1.0, "topic": 0.9, "summary": 0.6, "daily": 0.5, "session": 0.5}
# Alexander types names in Russian; transcripts and titles often carry them in Latin. The contacts
# directory already maps both spellings for the hook `contact-retrieval.py` — reuse it, fail-open.
CONTACTS_PATH = Path.home() / ".claude/projects/-Users-Syncopation-Documents-Git-Local/memory/contacts_directory.md"
RRF_K = 60                   # vector list
RRF_K_LEXICAL = 20           # exact matches are trusted more: rank 1 here (1/21) outweighs vector rank 1 (1/61)
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

        # Hybrid: fuse the vector ranking with an exact-token ranking (see TOKEN_RE note above).
        lexical = self._lexical_hits(query, where_filter)
        return self._fuse(collapsed[:30], lexical, n_results)

    # ------------------------------------------------------------------ lexical pass

    @staticmethod
    def query_tokens(query: str) -> list[str]:
        """Substrings worth an exact lookup. Cyrillic words longer than five characters are cut by
        two — a crude stem so «Сергеем» finds «Сергей» and «воронку» finds «воронка». Latin words and
        all-caps terms (SLA, PRD) are used as typed; `$contains` is case-sensitive, which is fine
        for proper nouns and hurts only for sentence-initial common words the IDF cap drops anyway."""
        seen: list[str] = []
        for tok in TOKEN_RE.findall(query):
            if len(tok) < 4 and not tok.isupper():
                continue
            if CYRILLIC_RE.match(tok) and len(tok) > 5:
                tok = tok[:-2]
            if tok not in seen:
                seen.append(tok)
        seen.sort(key=len, reverse=True)
        seen = seen[:LEXICAL_MAX_TOKENS]
        for extra in MemorySearcher.alias_expansions(seen):
            if extra not in seen:
                seen.append(extra)
        return seen

    _alias_groups: list[list[str]] | None = None

    @classmethod
    def alias_expansions(cls, tokens: list[str]) -> list[str]:
        """Other spellings of a name from contacts_directory.md § Aliases («Патрик» ↔ Patrick ↔
        Burkhalter). Matching is case-insensitive and accepts the crude stem as a prefix."""
        if cls._alias_groups is None:
            groups: list[list[str]] = []
            try:
                text = CONTACTS_PATH.read_text(encoding="utf-8")
                block = text.split("## Aliases", 1)[1].split("\n## ", 1)[0]
                for line in block.splitlines():
                    names = re.findall(r"`([^`]+)`", line)
                    if len(names) >= 2:
                        groups.append(names)
            except Exception:
                pass
            cls._alias_groups = groups
        out: list[str] = []
        for tok in tokens:
            low = tok.lower()
            for group in cls._alias_groups:
                if any(n.lower() == low or n.lower().startswith(low) for n in group):
                    out.extend(n for n in group if len(n) >= 4 and n.lower() != low)
        return out

    def _lexical_hits(self, query: str, where_filter: dict | None) -> list[dict]:
        """Files ranked by exact-token evidence: Σ over matched tokens of (1 / files the token hits)
        × min(chunks in this file, 5). Rare tokens dominate; common ones are dropped outright."""
        file_score: dict[str, float] = defaultdict(float)
        file_tokens: dict[str, set] = defaultdict(set)
        file_sample: dict[str, tuple] = {}
        for tok in self.query_tokens(query):
            try:
                got = self.collection.get(
                    where_document={"$contains": tok},
                    where=where_filter,
                    include=["metadatas"],
                    limit=LEXICAL_CHUNK_LIMIT,
                )
            except Exception:
                continue
            ids = got.get("ids") or []
            if not ids or len(ids) >= LEXICAL_CHUNK_LIMIT:
                continue  # nothing, or too common to mean anything
            per_file: Counter = Counter()
            for doc_id, meta in zip(ids, got["metadatas"]):
                key = meta.get("parent_id") or meta.get("filename", doc_id)
                per_file[key] += 1
                file_sample.setdefault(key, (doc_id, meta))
            if len(per_file) > LEXICAL_MAX_FILES:
                continue
            weight = 1.0 / len(per_file)
            for key, count in per_file.items():
                file_score[key] += weight * (1 + min(count, 5) / 5)
                file_tokens[key].add(tok)
        for key in file_score:
            source = file_sample[key][1].get("source_type", "")
            matched = len(file_tokens[key])
            file_score[key] *= (1 + 0.5 * (matched - 1)) * LEXICAL_SOURCE_WEIGHT.get(source, 0.7)
        ranked = sorted(file_score.items(), key=lambda kv: kv[1], reverse=True)
        return [
            {"key": key, "lexical_score": score, "tokens": sorted(file_tokens[key]), "sample": file_sample[key]}
            for key, score in ranked[:30]
        ]

    def _fuse(self, vector: list[dict], lexical: list[dict], n_results: int) -> list[dict]:
        """Reciprocal rank fusion: score = Σ 1/(k + rank) over the lists a file appears in, with a
        smaller k for the lexical list and each lexical contribution scaled by its share of the top
        lexical score. Plain RRF let five weak double matches outrank the one page that matched all
        three exact tokens (Kira × Serge funnel/SLA case, 2026-09-05); exact evidence now leads."""
        rrf: dict[str, float] = defaultdict(float)
        by_key: dict[str, dict] = {}
        for rank, r in enumerate(vector, 1):
            key = r.get("parent_id") or r["filename"]
            rrf[key] += 1.0 / (RRF_K + rank)
            r["lexical_tokens"] = []
            by_key[key] = r
        top_lex = lexical[0]["lexical_score"] if lexical else 1.0
        for rank, h in enumerate(lexical, 1):
            key = h["key"]
            rrf[key] += (h["lexical_score"] / top_lex) / (RRF_K_LEXICAL + rank)
            if key in by_key:
                by_key[key]["lexical_tokens"] = h["tokens"]
            else:
                by_key[key] = self._lexical_only_result(h)
        for key, r in by_key.items():
            r["rrf"] = rrf[key]
        return sorted(by_key.values(), key=lambda x: x["rrf"], reverse=True)[:n_results]

    def _lexical_only_result(self, hit: dict) -> dict:
        doc_id, metadata = hit["sample"]
        preview = ""
        try:
            got = self.collection.get(ids=[doc_id], include=["documents"])
            doc = (got.get("documents") or [""])[0]
            preview = "\n".join(l for l in doc.split("\n") if l.strip())[:600]
        except Exception:
            pass
        return {
            "doc_id": doc_id,
            "filename": metadata.get("filename", "unknown"),
            "title": metadata.get("title", "Untitled"),
            "date": metadata.get("session_date", "unknown"),
            "source_type": metadata.get("source_type", "unknown"),
            "similarity": 0.0,
            "hybrid_score": 0.0,
            "complexity": metadata.get("complexity", "unknown"),
            "technologies": json.loads(metadata.get("technologies", "[]")),
            "preview": preview,
            "memory_type": metadata.get("memory_type", ""),
            "description": metadata.get("description", ""),
            "parent_id": metadata.get("parent_id", ""),
            "chunk_index": metadata.get("chunk_index", 0),
            "chunks_matched": 0,
            "lexical_tokens": hit["tokens"],
        }

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
            if r.get("lexical_tokens"):
                content += f"**Exact:** {', '.join(r['lexical_tokens'])}\n"
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
            exact = f" | Exact: {', '.join(r['lexical_tokens'])}" if r.get("lexical_tokens") else ""
            print(f"Relevance: {r['hybrid_score']:.0%} | Similarity: {r['similarity']:.0%}{exact}")
            if r["description"]:
                print(f"Description: {r['description']}")
            print(f"\n{r['preview'][:400]}\n")


RETRIEVAL_LOG = Path(__file__).parent.parent / "logs" / "retrievals.jsonl"


def log_retrieval(query: str, source_filter: str | None, caller: str, results: list, elapsed_s: float | None = None) -> None:
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
                    "exact": r.get("lexical_tokens", []),
                }
                for r in results[:3]
            ],
            "n": len(results),
            "elapsed_s": round(elapsed_s, 2) if elapsed_s is not None else None,
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

    t0 = time.monotonic()
    searcher = MemorySearcher()
    results = searcher.search(query, n_results=3, source_filter=source_filter)
    log_retrieval(query, source_filter, caller, results, elapsed_s=time.monotonic() - t0)

    if compact:
        searcher.display_compact(results, query)
    else:
        searcher.display_results(results, query)


if __name__ == "__main__":
    main()
