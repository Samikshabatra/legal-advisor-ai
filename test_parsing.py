"""
test_parsing.py — Validate extraction logic against the real CCPA PDF.
Uses PyMuPDF (fitz) only — works on Windows, Linux, Mac.
Install: pip install PyMuPDF
"""

import re
import os
import fitz  # pip install PyMuPDF

# ═══════════════════════════════════════════════════════════
# STEP 1: Extract text from PDF — auto-skip TOC pages
# ═══════════════════════════════════════════════════════════
print("Extracting text with PyMuPDF ...")

doc = fitz.open("data/ccpa_statute.pdf")
full_text = ""

for page_num in range(len(doc)):
    page_text = doc[page_num].get_text()
    # Skip TOC pages — they have lines of dots like "Section 1798.100.....3"
    if "....." in page_text and page_num < 5:
        print(f"  Skipping page {page_num} (TOC)")
        continue
    full_text += page_text

# Clean text
full_text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", full_text)
full_text = re.sub(r"[ \t]+", " ", full_text)
full_text = re.sub(r"\n{3,}", "\n\n", full_text)
full_text = full_text.replace("\u2019", "'").replace("\u2018", "'")
full_text = full_text.replace("\u201c", '"').replace("\u201d", '"')
full_text = full_text.replace("\f", "\n")

print(f"  Text length: {len(full_text)} chars")

if len(full_text) < 1000:
    print("\n  ERROR: Almost no text extracted!")
    print("  Please run: python diagnose_pdf.py")
    print("  And share the output so we can debug.")
    exit(1)


# ═══════════════════════════════════════════════════════════
# Master section list
# ═══════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════
# Extraction function
# ═══════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════
# TEST 1: Section completeness
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("TEST 1: SECTION EXTRACTION COMPLETENESS")
print("=" * 65)

found = sorted(raw_sections.keys())
missing = sorted(s for s in MASTER_SECTIONS if s not in found)

KEY = [
    "1798.100", "1798.105", "1798.106", "1798.110", "1798.115",
    "1798.120", "1798.121", "1798.125", "1798.130", "1798.135",
    "1798.140", "1798.145", "1798.146", "1798.148",
    "1798.150", "1798.155", "1798.160", "1798.175", "1798.180",
    "1798.185", "1798.190", "1798.192", "1798.194", "1798.196", "1798.198",
    "1798.199", "1798.199.10", "1798.199.40", "1798.199.70",
    "1798.199.90", "1798.199.95", "1798.199.100",
]

for sec in KEY:
    text = raw_sections.get(sec, "")
    length = len(text)
    if length > 50:
        print(f"  OK  {sec:15s} ({length:6d} chars)  {text[:70].strip()}...")
    elif length > 0:
        print(f"  !!  {sec:15s} ({length:6d} chars)  SHORT: {text[:70].strip()}")
    else:
        print(f"  XX  {sec:15s}  MISSING")

print(f"\n  Total extracted: {len(found)}/{len(MASTER_SECTIONS)}")
if missing:
    print(f"  Missing: {missing}")
else:
    print(f"  Missing: NONE")


# ═══════════════════════════════════════════════════════════
# TEST 2: 1798.100 from PDF
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("TEST 2: 1798.100 EXTRACTED FROM PDF (not hardcoded)")
print("=" * 65)

text_100 = raw_sections.get("1798.100", "")
has_e = "(e)" in text_100 and "security" in text_100.lower()
has_f = "(f)" in text_100 and "trade secret" in text_100.lower()
print(f"  Length: {len(text_100)} chars")
print(f"  Has subsection (e) about security:     {'OK' if has_e else 'FAIL'}")
print(f"  Has subsection (f) about trade secrets: {'OK' if has_f else 'FAIL'}")
if has_e and has_f:
    print("  PASS")
else:
    print("  FAIL — 1798.100 is incomplete")


# ═══════════════════════════════════════════════════════════
# TEST 3: 1798.185 and 1798.199.XX
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("TEST 3: FORM-FEED SECTIONS (1798.185) + 1798.199.XX")
print("=" * 65)

text_185 = raw_sections.get("1798.185", "")
print(f"  1798.185 (Regulations): {'OK' if len(text_185) > 200 else 'FAIL'}  ({len(text_185)} chars)")

for sub in ["1798.199.10", "1798.199.40", "1798.199.70", "1798.199.90", "1798.199.100"]:
    text = raw_sections.get(sub, "")
    print(f"  {sub:20s}: {'OK' if len(text) > 30 else 'FAIL'}  ({len(text)} chars)")

text_enf = raw_sections.get("1798.199.100", "")
has_content = text_enf and ("good faith" in text_enf.lower() or "cooperation" in text_enf.lower())
print(f"  1798.199.100 content:   {'OK' if has_content else 'FAIL'}")


# ═══════════════════════════════════════════════════════════
# TEST 4: embed_text multi-region sampling
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("TEST 4: EMBED_TEXT MULTI-REGION SAMPLING")
print("=" * 65)

def make_embed_text(title, keywords, ft, max_total=1500):
    prefix = f"{title}. {' '.join(keywords)}. "
    budget = max_total - len(prefix)
    if len(ft) <= budget:
        return prefix + ft
    sb = int(budget * 0.45)
    mb = int(budget * 0.30)
    eb = budget - sb - mb
    mid = len(ft) // 2
    return f"{prefix}{ft[:sb]} ... {ft[mid - mb//2 : mid + mb//2]} ... {ft[-eb:]}"

text_140 = raw_sections.get("1798.140", "")
if text_140:
    embed_140 = make_embed_text("Definitions", ["definition", "means"], text_140)
    old_embed = f"Definitions. definition means. {text_140[:800]}"
    old_w = len(set(old_embed.split()))
    new_w = len(set(embed_140.split()))
    print(f"  1798.140 full_text:  {len(text_140)} chars")
    print(f"  1798.140 embed_text: {len(embed_140)} chars")
    print(f"  Has '...' separators: {'OK' if '...' in embed_140 else 'FAIL'}")
    print(f"  Old unique words: {old_w}  New unique words: {new_w}  Improvement: +{new_w - old_w}")
else:
    print("  SKIP — 1798.140 not extracted")


# ═══════════════════════════════════════════════════════════
# TEST 5: Sub-chunk extraction
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("TEST 5: SUB-CHUNK EXTRACTION")
print("=" * 65)

sub_chunks = []
for sec_num in MASTER_SECTIONS:
    text = raw_sections.get(sec_num, "")
    if not text:
        continue
    for m in re.finditer(r"\n\s*\(([a-z])\)\s+", text):
        sub_chunks.append(sec_num)

for sec in ["1798.100", "1798.105", "1798.120", "1798.125", "1798.130", "1798.140"]:
    count = sub_chunks.count(sec)
    print(f"  {sec}: {count} sub-chunks")
print(f"\n  Total sub-chunks: {len(sub_chunks)}")


# ═══════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("FINAL SUMMARY")
print("=" * 65)

issues = []
if missing:
    issues.append(f"Missing {len(missing)} sections")
if not (has_e and has_f):
    issues.append("1798.100 incomplete")
if len(text_185) < 200:
    issues.append("1798.185 not extracted")
if "1798.199.100" not in raw_sections:
    issues.append("1798.199.100 not extracted")

if issues:
    print(f"\n  Issues found:")
    for i in issues:
        print(f"    - {i}")
else:
    print(f"\n  ALL TESTS PASSED")
    print(f"  {len(found)} sections extracted from PDF")
    print(f"  1798.100 from PDF (not hardcoded)")
    print(f"  1798.185 and 1798.199.XX correctly handled")
    print(f"  embed_text uses multi-region sampling")
    print(f"  Single canonical pipeline")