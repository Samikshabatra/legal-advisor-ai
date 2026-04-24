# build_database.py — Fixed version
# Fixes:
#   1. 1798.100 extracted from PDF (not hardcoded)
#   2. Robust regex — handles form-feeds, cross-references, 1798.199.XX
#   3. embed_text uses multi-region sampling (not 800-char truncation)
#   4. Single canonical pipeline
#   5. Auto-detects TOC pages (works on any system)

import fitz  # pip install PyMuPDF
import re
import json
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer

os.makedirs("data", exist_ok=True)

# ════════════════════════════════════════════════════════════
# STEP 1: Extract text — auto-skip TOC pages
# ════════════════════════════════════════════════════════════
print("Step 1: Extracting text from PDF ...")

doc = fitz.open("data/ccpa_statute.pdf")
full_text = ""
for page_num in range(len(doc)):
    page_text = doc[page_num].get_text()
    # Skip TOC pages — they contain lines of dots like "1798.100......3"
    if "....." in page_text and page_num < 5:
        print(f"  Skipping page {page_num} (TOC)")
        continue
    full_text += page_text

# Clean
full_text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", full_text)
full_text = re.sub(r"[ \t]+", " ", full_text)
full_text = re.sub(r"\n{3,}", "\n\n", full_text)
full_text = full_text.replace("\u2019", "'").replace("\u2018", "'")
full_text = full_text.replace("\u201c", '"').replace("\u201d", '"')
full_text = full_text.replace("\f", "\n")

print(f"  Extracted {len(full_text)} characters")

if len(full_text) < 1000:
    print("  ERROR: Almost no text extracted!")
    print("  Run: python diagnose_pdf.py  to debug")
    exit(1)


# ════════════════════════════════════════════════════════════
# STEP 2: Master section list
# ════════════════════════════════════════════════════════════
print("\nStep 2: Master section list ...")

MASTER_SECTIONS = {
    "1798.100":     "General Duties of Businesses that Collect Personal Information",
    "1798.105":     "Consumers Right to Delete Personal Information",
    "1798.106":     "Consumers Right to Correct Inaccurate Personal Information",
    "1798.110":     "Consumers Right to Know What Personal Information is Being Collected",
    "1798.115":     "Consumers Right to Know What Personal Information is Sold or Shared",
    "1798.120":     "Consumers Right to Opt Out of Sale or Sharing of Personal Information",
    "1798.121":     "Consumers Right to Limit Use and Disclosure of Sensitive Personal Information",
    "1798.125":     "Consumers Right of No Retaliation Following Opt Out or Exercise of Other Rights",
    "1798.130":     "Notice Disclosure Correction and Deletion Requirements",
    "1798.135":     "Methods of Limiting Sale Sharing and Use of Personal Information",
    "1798.140":     "Definitions",
    "1798.145":     "Exemptions",
    "1798.146":     "Additional Exemptions",
    "1798.148":     "Reidentification Restrictions",
    "1798.150":     "Personal Information Security Breaches",
    "1798.155":     "Administrative Enforcement",
    "1798.160":     "Consumer Privacy Fund",
    "1798.175":     "Conflicting Provisions",
    "1798.180":     "Preemption",
    "1798.185":     "Regulations",
    "1798.190":     "Anti-Avoidance",
    "1798.192":     "Waiver",
    "1798.194":     "Liberal Construction",
    "1798.196":     "Supplement to Federal and State Law",
    "1798.198":     "Operative Date",
    "1798.199":     "Operative Date of Section 1798.180",
    "1798.199.10":  "California Privacy Protection Agency Establishment",
    "1798.199.15":  "Agency Board Member Duties",
    "1798.199.20":  "Board Member Terms",
    "1798.199.25":  "Board Member Compensation",
    "1798.199.30":  "Executive Director",
    "1798.199.35":  "Delegation of Authority",
    "1798.199.40":  "Agency Functions",
    "1798.199.45":  "Agency Rulemaking and Guidelines",
    "1798.199.50":  "Probable Cause Findings",
    "1798.199.55":  "Administrative Enforcement Proceedings",
    "1798.199.60":  "Rejection of ALJ Decision",
    "1798.199.65":  "Subpoena Power",
    "1798.199.70":  "Statute of Limitations",
    "1798.199.75":  "Civil Penalties",
    "1798.199.80":  "Collection of Administrative Fines",
    "1798.199.85":  "Judicial Review",
    "1798.199.90":  "Administrative Fines",
    "1798.199.95":  "Agency Appropriation",
    "1798.199.100": "Good Faith Consideration",
}

SECTION_KEYWORDS = {
    "1798.100": ["collect", "collection", "inform", "notice", "purpose", "disclose",
                 "categories", "privacy policy"],
    "1798.105": ["delete", "deletion", "remove", "erase", "request", "right to delete"],
    "1798.106": ["correct", "correction", "inaccurate", "accurate", "rectify"],
    "1798.110": ["know", "access", "categories", "collected", "right to know", "what data"],
    "1798.115": ["sold", "shared", "third party", "disclosed", "whom", "to whom"],
    "1798.120": ["opt out", "opt-out", "sale", "sell", "share", "sharing",
                 "do not sell", "minor", "child", "16", "13"],
    "1798.121": ["sensitive", "limit", "restrict", "health", "financial",
                 "precise location", "biometric", "race", "religion"],
    "1798.125": ["discriminate", "discrimination", "retaliation", "price", "penalty",
                 "different price", "deny", "level of service"],
    "1798.130": ["notice", "respond", "response", "45 days", "verify", "verification",
                 "method", "disclosure requirement", "toll-free", "format"],
    "1798.135": ["link", "opt-out link", "opt out link", "do not sell",
                 "homepage", "methods", "limiting", "alternative opt-out"],
    "1798.140": ["definition", "means", "business", "personal information", "consumer",
                 "service provider", "contractor", "sell", "share",
                 "sensitive personal information", "aggregate"],
    "1798.145": ["exempt", "exemption", "not apply", "exception", "HIPAA",
                 "GLBA", "FCRA", "employee"],
    "1798.146": ["exempt", "medical", "clinical trial", "HIPAA"],
    "1798.148": ["reidentify", "deidentified"],
    "1798.150": ["private right", "action", "damages", "breach", "unauthorized",
                 "security", "data breach"],
    "1798.155": ["attorney general", "enforce", "penalty", "fine", "violation",
                 "civil penalty", "agency enforcement"],
    "1798.160": ["fund", "revenue", "consumer privacy fund"],
    "1798.175": ["relationship", "federal", "other laws", "conflicting"],
    "1798.180": ["preempt", "local", "ordinance", "statewide"],
    "1798.185": ["regulation", "rulemaking", "implement", "attorney general",
                 "adopt regulations"],
    "1798.190": ["circumvent", "avoid", "anti-avoidance"],
    "1798.192": ["waiver", "waive", "contract", "void"],
    "1798.194": ["construe", "construction", "liberal"],
    "1798.196": ["supplement", "federal law", "preempted"],
    "1798.198": ["operative", "effective date"],
    "1798.199":     ["operative", "1798.180"],
    "1798.199.10":  ["agency", "privacy protection agency", "board", "establish"],
    "1798.199.40":  ["agency functions", "investigate", "audit", "enforce"],
    "1798.199.55":  ["administrative enforcement", "cease and desist", "hearing"],
    "1798.199.70":  ["statute of limitations", "five years"],
    "1798.199.75":  ["civil penalty", "penalty amount"],
    "1798.199.90":  ["administrative fine", "fine amount"],
    "1798.199.95":  ["appropriation", "budget", "funding"],
    "1798.199.100": ["good faith", "cooperation"],
}

print(f"  {len(MASTER_SECTIONS)} sections in master list")


# ════════════════════════════════════════════════════════════
# STEP 3: Extract full text for each section
# ════════════════════════════════════════════════════════════
print("\nStep 3: Extracting full section texts ...")

def extract_full_sections(text, master_sections):
    sections = {}
    positions = []

    sorted_nums = sorted(master_sections.keys(), key=lambda x: (-len(x), x))

    for sec_num in sorted_nums:
        escaped = re.escape(sec_num)
        pattern = re.compile(
            rf"(?:^|\n)\s*{escaped}\.?\s",
            re.MULTILINE
        )
        for match in pattern.finditer(text):
            pos = match.start()
            context = text[pos:pos + 200]
            if "....." in context:
                continue

            # Skip cross-references like "pursuant to Section 1798.185"
            before = text[max(0, pos - 80):pos].rstrip()
            if before.endswith("Section") or before.endswith("of Section"):
                continue
            if re.search(r"(?:Sections?|section|of)\s*$", before, re.IGNORECASE):
                continue
            if re.search(r"(?:and|,)\s*$", before):
                continue

            if any(abs(pos - p) < 20 for p, _ in positions):
                continue
            positions.append((pos, sec_num))

    positions.sort(key=lambda x: x[0])

    # Dedup: keep LAST occurrence per section
    seen = {}
    for pos, sec_num in positions:
        seen[sec_num] = (pos, sec_num)
    positions = sorted(seen.values(), key=lambda x: x[0])
    print(f"  Found {len(positions)} unique section positions")

    for i, (start_pos, sec_num) in enumerate(positions):
        end_pos = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        raw_text = text[start_pos:end_pos].strip()

        header_re = re.compile(
            rf"^{re.escape(sec_num)}\.?\s+[^\n]*\n",
            re.MULTILINE
        )
        m = header_re.match(raw_text)
        if m:
            raw_text = raw_text[m.end():].strip()
        else:
            raw_text = re.sub(rf"^{re.escape(sec_num)}\.?\s*", "", raw_text).strip()

        sections[sec_num] = raw_text

    return sections


raw_sections = extract_full_sections(full_text, MASTER_SECTIONS)

found = sorted(raw_sections.keys())
missing = sorted(s for s in MASTER_SECTIONS if s not in found)
print(f"  Extracted: {len(found)} sections")
if missing:
    print(f"  Missing: {missing}")


# ════════════════════════════════════════════════════════════
# STEP 4: Extract definitions from 1798.140
# ════════════════════════════════════════════════════════════
print("\nStep 4: Extracting definitions ...")

def extract_definitions(definitions_text):
    entries = []
    def_pattern = re.compile(
        r'\(([a-z])\)\s+"?([^"\n]+?)"?\s+means?\s+',
        re.MULTILINE | re.IGNORECASE
    )
    matches = list(def_pattern.finditer(definitions_text))
    for i, match in enumerate(matches):
        sub = match.group(1)
        term = match.group(2).strip().strip('"').strip("'")
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(definitions_text)
        def_text = definitions_text[start:end].strip()
        entries.append({"term": term, "subsection": sub, "text": def_text})
    print(f"  Found {len(entries)} definition terms")
    return entries

definitions_entries = []
if "1798.140" in raw_sections:
    definitions_entries = extract_definitions(raw_sections["1798.140"])


# ════════════════════════════════════════════════════════════
# STEP 5: Build structured records
# ════════════════════════════════════════════════════════════
print("\nStep 5: Building structured records ...")

def make_embed_text(title, keywords, full_text, max_total=1500):
    prefix = f"{title}. {' '.join(keywords)}. "
    budget = max_total - len(prefix)
    if len(full_text) <= budget:
        return prefix + full_text
    sb = int(budget * 0.45)
    mb = int(budget * 0.30)
    eb = budget - sb - mb
    mid = len(full_text) // 2
    return f"{prefix}{full_text[:sb]} ... {full_text[mid - mb//2 : mid + mb//2]} ... {full_text[-eb:]}"

# Full sections
full_section_records = []
for sec_num, title in MASTER_SECTIONS.items():
    text = raw_sections.get(sec_num, "")
    if not text:
        continue
    keywords = SECTION_KEYWORDS.get(sec_num, [])
    full_section_records.append({
        "section_number": sec_num,
        "section":        f"Section {sec_num}",
        "title":          title,
        "keywords":       keywords,
        "full_text":      text,
        "embed_text":     make_embed_text(title, keywords, text),
    })
print(f"  Full section records: {len(full_section_records)}")

# Sub-chunks
sub_chunk_records = []
for sec_num, title in MASTER_SECTIONS.items():
    text = raw_sections.get(sec_num, "")
    if not text:
        continue
    sub_pattern = re.compile(r"\n\s*\(([a-z])\)\s+", re.MULTILINE)
    sub_matches = list(sub_pattern.finditer(text))
    if not sub_matches:
        continue
    for i, match in enumerate(sub_matches):
        sub_letter = match.group(1)
        start = match.end()
        end = sub_matches[i + 1].start() if i + 1 < len(sub_matches) else len(text)
        sub_text = text[start:end].strip()
        if len(sub_text) > 50:
            sub_chunk_records.append({
                "section_number": sec_num,
                "section":        f"Section {sec_num}",
                "subsection":     sub_letter,
                "title":          title,
                "chunk_text":     f"{title} - ({sub_letter}). {sub_text}",
            })

for defn in definitions_entries:
    sub_chunk_records.append({
        "section_number": "1798.140",
        "section":        "Section 1798.140",
        "subsection":     defn["subsection"],
        "title":          f"Definition of {defn['term']}",
        "chunk_text":     f"Definition: {defn['term']} - {defn['text']}",
    })

print(f"  Sub-chunk records:    {len(sub_chunk_records)}")


# ════════════════════════════════════════════════════════════
# STEP 6: Save JSON files
# ════════════════════════════════════════════════════════════
print("\nStep 6: Saving databases ...")

with open("data/ccpa_full_sections.json", "w", encoding="utf-8") as f:
    json.dump(full_section_records, f, indent=2)
print("  Saved data/ccpa_full_sections.json")

with open("data/ccpa_sub_chunks.json", "w", encoding="utf-8") as f:
    json.dump(sub_chunk_records, f, indent=2)
print("  Saved data/ccpa_sub_chunks.json")

valid_sections = list(MASTER_SECTIONS.keys())
with open("data/valid_sections.json", "w", encoding="utf-8") as f:
    json.dump(valid_sections, f, indent=2)
print("  Saved data/valid_sections.json")

# Remove stale files from old pipeline
for stale in ["data/ccpa_sections.json", "data/ccpa.index"]:
    if os.path.exists(stale):
        os.remove(stale)
        print(f"  Removed stale {stale}")


# ════════════════════════════════════════════════════════════
# STEP 7: Build FAISS indexes
# ════════════════════════════════════════════════════════════
print("\nStep 7: Building embeddings + FAISS indexes ...")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("  Embedding full sections ...")
full_texts = [r["embed_text"] for r in full_section_records]
full_embeddings = embedder.encode(full_texts, show_progress_bar=True)
full_embeddings = np.array(full_embeddings).astype("float32")
index_full = faiss.IndexFlatL2(full_embeddings.shape[1])
index_full.add(full_embeddings)
faiss.write_index(index_full, "data/ccpa_full.index")
print(f"  Index A (full): {index_full.ntotal} vectors")

print("  Embedding sub-chunks ...")
sub_texts = [r["chunk_text"] for r in sub_chunk_records]
sub_embeddings = embedder.encode(sub_texts, show_progress_bar=True)
sub_embeddings = np.array(sub_embeddings).astype("float32")
index_sub = faiss.IndexFlatL2(sub_embeddings.shape[1])
index_sub.add(sub_embeddings)
faiss.write_index(index_sub, "data/ccpa_sub.index")
print(f"  Index B (sub):  {index_sub.ntotal} vectors")


# ════════════════════════════════════════════════════════════
# STEP 8: Quick search test
# ════════════════════════════════════════════════════════════
print("\nStep 8: Testing hybrid search ...")

def hybrid_search(query, top_k=5):
    query_lower = query.lower()
    vec = embedder.encode([query]).astype("float32")
    _, sub_idxs  = index_sub.search(vec, top_k * 2)
    _, full_idxs = index_full.search(vec, top_k)
    candidates = {}
    for rank, idx in enumerate(sub_idxs[0]):
        sec = sub_chunk_records[idx]["section_number"]
        candidates[sec] = candidates.get(sec, 0) + (top_k * 2 - rank)
    for rank, idx in enumerate(full_idxs[0]):
        sec = full_section_records[idx]["section_number"]
        candidates[sec] = candidates.get(sec, 0) + (top_k - rank)
    for sec_num, keywords in SECTION_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in query_lower:
                candidates[sec_num] = candidates.get(sec_num, 0) + 3
                break
    ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return [r[0] for r in ranked[:top_k]]

tests = [
    ("We sell customer data to ad networks without telling them", ["1798.120", "1798.115"]),
    ("We ignored a user's request to delete their data",         ["1798.105"]),
    ("We charge higher prices to users who opt out",             ["1798.125"]),
    ("We collect data of 14-year-olds without parental consent", ["1798.120"]),
    ("We have no privacy notice on our website",                 ["1798.100", "1798.130"]),
]

correct = 0
for prompt, expected in tests:
    results = hybrid_search(prompt, top_k=5)
    hits = [e for e in expected if e in results]
    status = "OK" if len(hits) == len(expected) else "PARTIAL"
    if len(hits) == len(expected):
        correct += 1
    print(f"  {status}  '{prompt[:55]}...'")
    print(f"       Expected: {expected}  Retrieved: {results[:5]}")

print(f"\n  Score: {correct}/{len(tests)} fully correct")


# ════════════════════════════════════════════════════════════
# DONE
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("BUILD COMPLETE!")
print(f"  Full sections: {len(full_section_records)}")
print(f"  Sub-chunks:    {len(sub_chunk_records)}")
print(f"  FAISS indexes: ccpa_full.index + ccpa_sub.index")
print("=" * 55)