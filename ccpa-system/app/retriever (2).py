import json
import numpy as np
import faiss
import os
import re
from sentence_transformers import SentenceTransformer


class CCPARetriever:
    def __init__(self, data_dir="data/"):
        """Load all data and indexes at startup (once)."""
        print("[Retriever] Loading data...")

        # Load JSON databases
        with open(os.path.join(data_dir, "ccpa_full_sections.json"), "r", encoding="utf-8") as f:
            self.full_sections = json.load(f)

        with open(os.path.join(data_dir, "ccpa_sub_chunks.json"), "r", encoding="utf-8") as f:
            self.sub_chunks = json.load(f)

        with open(os.path.join(data_dir, "valid_sections.json"), "r", encoding="utf-8") as f:
            self.valid_sections = set(json.load(f))

        # Build lookup maps
        self.section_map = {s["section_number"]: s for s in self.full_sections}

        # Build keyword index from full_sections
        self.section_keywords = {}
        for s in self.full_sections:
            self.section_keywords[s["section_number"]] = [
                kw.lower() for kw in s.get("keywords", [])
            ]

        # Load embedding model
        print("[Retriever] Loading embedding model...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # Load or build FAISS indexes
        full_index_path = os.path.join(data_dir, "ccpa_full.index")
        sub_index_path = os.path.join(data_dir, "ccpa_sub.index")

        if os.path.exists(full_index_path) and os.path.exists(sub_index_path):
            print("[Retriever] Loading FAISS indexes from disk...")
            self.index_full = faiss.read_index(full_index_path)
            self.index_sub = faiss.read_index(sub_index_path)
        else:
            print("[Retriever] Building FAISS indexes (first run)...")
            self._build_indexes(data_dir)

        print(f"[Retriever] Ready — {len(self.full_sections)} sections, "
              f"{len(self.sub_chunks)} sub-chunks, "
              f"{self.index_full.ntotal} full vectors, "
              f"{self.index_sub.ntotal} sub vectors")

    def _build_indexes(self, data_dir):
        """Build FAISS indexes from scratch and save to disk."""
        # Full sections index
        full_texts = [s["embed_text"] for s in self.full_sections]
        full_embs = self.embedder.encode(full_texts, show_progress_bar=True)
        full_embs = np.array(full_embs).astype("float32")
        self.index_full = faiss.IndexFlatIP(full_embs.shape[1])  # Inner product (cosine-like)
        faiss.normalize_L2(full_embs)  # Normalize for cosine similarity
        self.index_full.add(full_embs)
        faiss.write_index(self.index_full, os.path.join(data_dir, "ccpa_full.index"))

        # Sub-chunks index
        sub_texts = [s["chunk_text"] for s in self.sub_chunks]
        sub_embs = self.embedder.encode(sub_texts, show_progress_bar=True)
        sub_embs = np.array(sub_embs).astype("float32")
        self.index_sub = faiss.IndexFlatIP(sub_embs.shape[1])
        faiss.normalize_L2(sub_embs)
        self.index_sub.add(sub_embs)
        faiss.write_index(self.index_sub, os.path.join(data_dir, "ccpa_sub.index"))

        print(f"[Retriever] Built indexes: {self.index_full.ntotal} full, {self.index_sub.ntotal} sub")

    def _keyword_scores(self, query: str) -> dict:
        """
        Score sections by keyword matches in the query.
        Returns {section_number: score}.

        Scoring:
          - Each keyword match: +3 points
          - Multi-word keyword match (more specific): +5 points
          - Cap at +15 per section to prevent keyword domination
        """
        query_lower = query.lower()
        scores = {}

        for sec_num, keywords in self.section_keywords.items():
            sec_score = 0
            for kw in keywords:
                if kw in query_lower:
                    # Multi-word keywords are more specific → higher score
                    if " " in kw:
                        sec_score += 5
                    else:
                        sec_score += 3
            # Cap keyword boost to prevent one section from dominating
            scores[sec_num] = min(sec_score, 15)

        return scores

    def _vector_search(self, query: str, top_k: int = 5) -> dict:
        """
        Run vector similarity search on both indexes.
        Returns {section_number: score}.

        Sub-chunks searched with 2x top_k for better recall,
        then scores aggregated by parent section.
        """
        # Encode query
        query_vec = self.embedder.encode([query])
        query_vec = np.array(query_vec).astype("float32")
        faiss.normalize_L2(query_vec)

        scores = {}

        # Search sub-chunks (precise retrieval)
        sub_k = min(top_k * 3, len(self.sub_chunks))
        sub_dists, sub_idxs = self.index_sub.search(query_vec, sub_k)

        for rank, (dist, idx) in enumerate(zip(sub_dists[0], sub_idxs[0])):
            if idx < 0 or idx >= len(self.sub_chunks):
                continue
            sec_num = self.sub_chunks[idx]["section_number"]
            # Score: higher rank = higher score, weighted by similarity
            rank_score = (sub_k - rank) / sub_k  # 1.0 → 0.0
            sim_score = max(0, float(dist))  # cosine similarity (0-1)
            combined = rank_score * 5 + sim_score * 10
            scores[sec_num] = scores.get(sec_num, 0) + combined

        # Search full sections (broad retrieval)
        full_k = min(top_k * 2, len(self.full_sections))
        full_dists, full_idxs = self.index_full.search(query_vec, full_k)

        for rank, (dist, idx) in enumerate(zip(full_dists[0], full_idxs[0])):
            if idx < 0 or idx >= len(self.full_sections):
                continue
            sec_num = self.full_sections[idx]["section_number"]
            rank_score = (full_k - rank) / full_k
            sim_score = max(0, float(dist))
            combined = rank_score * 3 + sim_score * 7
            scores[sec_num] = scores.get(sec_num, 0) + combined

        return scores

    def retrieve(self, query: str, top_k: int = 5) -> list:
        """
        Main retrieval method. Returns top_k most relevant CCPA sections.

        Returns list of dicts:
        [
            {
                "section_number": "1798.120",
                "section": "Section 1798.120",
                "title": "Consumers Right to Opt Out...",
                "full_text": "...",
                "score": 25.3
            },
            ...
        ]
        """
        # Get scores from both methods
        vector_scores = self._vector_search(query, top_k)
        keyword_scores = self._keyword_scores(query)

        # Fuse scores
        all_sections = set(vector_scores.keys()) | set(keyword_scores.keys())
        fused = {}
        for sec in all_sections:
            fused[sec] = vector_scores.get(sec, 0) + keyword_scores.get(sec, 0)

        # Sort by score descending
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)

        # Build result list with full section data
        results = []
        for sec_num, score in ranked[:top_k]:
            if sec_num in self.section_map:
                record = self.section_map[sec_num]
                results.append({
                    "section_number": sec_num,
                    "section": record["section"],
                    "title": record["title"],
                    "full_text": record["full_text"],
                    "score": round(score, 2),
                })

        return results

    def get_section_text(self, section_number: str) -> str:
        """Get full text for a specific section number."""
        if section_number in self.section_map:
            return self.section_map[section_number]["full_text"]
        return ""

    def is_valid_section(self, section_number: str) -> bool:
        """Check if a section number is valid (exists in CCPA)."""
        return section_number in self.valid_sections

    def validate_citations(self, citations: list) -> list:
        """
        Filter a list of citation strings to only valid ones.
        Handles formats like "Section 1798.120" or just "1798.120".
        """
        valid = []
        for cite in citations:
            # Extract the number part
            num = cite.replace("Section ", "").replace("section ", "").strip()
            if num in self.valid_sections:
                valid.append(f"Section {num}")
        return valid


# ================================================================
# Standalone test
# ================================================================
if __name__ == "__main__":
    import time

    # Determine data directory
    data_dir = "data/"
    if not os.path.exists(os.path.join(data_dir, "ccpa_full_sections.json")):
        print("ERROR: data/ directory not found. Run from project root.")
        exit(1)

    print("=" * 60)
    print("RETRIEVER TEST")
    print("=" * 60)

    start = time.time()
    retriever = CCPARetriever(data_dir)
    load_time = time.time() - start
    print(f"\nLoad time: {load_time:.2f}s")

    # Hackathon test cases
    test_cases = [
        # (prompt, expected_sections, is_harmful, description)
        (
            "We are selling our customers' personal information to third-party data brokers without informing them or giving them a chance to opt out.",
            ["1798.120"],
            True,
            "Selling data without opt-out"
        ),
        (
            "Our company collects browsing history, geolocation, and biometric data from users but our privacy policy doesn't mention any of this.",
            ["1798.100"],
            True,
            "Undisclosed data collection"
        ),
        (
            "A customer asked us to delete their data but we are ignoring their request and keeping all records.",
            ["1798.105"],
            True,
            "Ignoring deletion request"
        ),
        (
            "We charge customers who opted out of data selling a higher price for the same service.",
            ["1798.125"],
            True,
            "Discriminatory pricing"
        ),
        (
            "We are collecting and selling personal data of 14-year-old users without getting their parent's consent.",
            ["1798.120"],
            True,
            "Minor's data without consent"
        ),
        (
            "Our company provides a clear privacy policy and allows customers to opt out of data selling at any time.",
            [],
            False,
            "Compliant practices"
        ),
        (
            "We deleted all personal data within 45 days after receiving the consumer's verified request.",
            [],
            False,
            "Proper deletion compliance"
        ),
        (
            "Can we schedule a team meeting for next Monday to discuss the project?",
            [],
            False,
            "Unrelated to CCPA"
        ),
        (
            "Our website has a 'Do Not Sell My Personal Information' link on the homepage as required.",
            [],
            False,
            "Proper opt-out link"
        ),
        (
            "We provide equal service and pricing to all customers regardless of whether they exercise their privacy rights.",
            [],
            False,
            "Non-discriminatory practices"
        ),
    ]

    print()
    correct = 0
    total_harmful = 0
    total_time = 0

    for prompt, expected, is_harmful, desc in test_cases:
        start = time.time()
        results = retriever.retrieve(prompt, top_k=5)
        elapsed = time.time() - start
        total_time += elapsed

        retrieved = [r["section_number"] for r in results]
        retrieved_scores = {r["section_number"]: r["score"] for r in results}

        if not expected:
            print(f"  🟢 SAFE — {desc}")
            print(f"     Top retrieved: {retrieved[:3]} (LLM decides final verdict)")
            print(f"     Time: {elapsed*1000:.0f}ms")
            print()
            continue

        total_harmful += 1
        hits = [e for e in expected if e in retrieved]
        misses = [e for e in expected if e not in retrieved]
        rate = len(hits) / len(expected)

        if rate == 1.0:
            status = "✅ PERFECT"
            correct += 1
        elif rate >= 0.5:
            status = "🟡 PARTIAL"
        else:
            status = "❌ MISSED"

        print(f"  {status} — {desc}")
        print(f"     Expected:  {expected}")
        print(f"     Retrieved: {retrieved[:5]}")
        if results:
            print(f"     Top scores: {[(r['section_number'], r['score']) for r in results[:3]]}")
        if misses:
            print(f"     Missed:    {misses}")
        print(f"     Time: {elapsed*1000:.0f}ms")
        print()

    print("=" * 60)
    print(f"  Harmful cases: {correct}/{total_harmful} fully correct")
    print(f"  Avg retrieval time: {total_time/len(test_cases)*1000:.0f}ms per query")
    print("=" * 60)