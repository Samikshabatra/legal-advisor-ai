"""
diagnose_pdf.py — Run this first to see your PDF's page structure.
Usage: python diagnose_pdf.py
"""
import fitz

doc = fitz.open("data/ccpa_statute.pdf")
print(f"Total pages: {len(doc)}")
print()

for page_num in range(min(len(doc), 5)):  # check first 5 pages
    page = doc[page_num]
    text = page.get_text()
    preview = text[:150].replace("\n", " ").strip()
    print(f"Page {page_num}: {len(text)} chars")
    print(f"  Preview: {preview}...")
    print()