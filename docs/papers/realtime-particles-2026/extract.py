"""Archive these references with Poppler; run from any working directory.

Usage: python3 extract.py [paper-id ...]
Existing PDFs are reused. Derived files are regenerated. Requires Poppler.
"""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parent
SOURCES = json.loads((ROOT / "sources.json").read_text())


def run(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def extract(paper):
    folder = ROOT / paper["id"]
    folder.mkdir(exist_ok=True)
    pdf = folder / "paper.pdf"
    if not pdf.exists():
        request = urllib.request.Request(paper["url"], headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read()
        if not data.startswith(b"%PDF-"):
            raise ValueError(f"Not a PDF: {paper['id']}")
        pdf.write_bytes(data)
    info = run("pdfinfo", str(pdf))
    pages = int(re.search(r"^Pages:\s+(\d+)", info, re.M)[1])
    run("pdftotext", "-layout", "-enc", "UTF-8", str(pdf), str(folder / "paper.txt"))
    embedded = folder / "embedded"
    embedded.mkdir(exist_ok=True)
    run("pdfimages", "-png", "-j", str(pdf), str(embedded / "image"))
    image_map = run("pdfimages", "-list", str(pdf))
    (folder / "embedded-images.txt").write_text(image_map)
    run("pdftoppm", "-png", "-r", "144", str(pdf), str(folder / "page"))
    renders = sorted(folder.glob("page-*.png"))
    if len(renders) != pages:
        raise ValueError(f"Page count mismatch: {paper['id']}")
    text_pages = (folder / "paper.txt").read_text().split("\f")[:pages]
    figures = []
    for i, content in enumerate(text_pages, 1):
        labels = list(dict.fromkeys(re.findall(r"\b(?:Fig(?:ure)?\.|Figure)\s+\d+[.:]?", content)))
        figures.append({"page": i, "render": renders[i-1].name, "figure_mentions": labels})
    record = {**paper, "accessed": SOURCES["accessed"], "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
              "pages": pages, "dpi": 144, "embedded_files": len(list(embedded.iterdir())),
              "page_map": figures}
    (folder / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    rows = "\n".join(f"| {p['page']} | [{p['render']}]({p['render']}) | {', '.join(p['figure_mentions']) or '-'} |" for p in figures)
    (folder / "README.md").write_text(
        f"# {paper['title']}\n\n"
        f"Published {paper['year']}; retrieved {SOURCES['accessed']}.\n\n"
        f"[Publisher or author page]({paper['project']}) · [Source PDF]({paper['url']}) · "
        "[Local PDF](paper.pdf) · [Extracted text](paper.txt)\n\n"
        f"PDF SHA-256: `{record['sha256']}`\n\n"
        f"All {pages} pages are rendered at 144 DPI, preserving vector diagrams, captions, equations and tables. "
        f"The `embedded/` directory contains {record['embedded_files']} native raster image and mask files. "
        "[Embedded image map](embedded-images.txt) records source page, image number, dimensions, encoding and masks. "
        "An embedded object may be only part of a figure; the complete page is the visual authority. "
        "Figure mentions below are automatically extracted and may include cross-references.\n\n"
        "| Page | Complete render | Figure mentions |\n| --- | --- | --- |\n" + rows + "\n\n"
        "Reproduce from the collection directory with `python3 extract.py " + paper['id'] + "`. "
        "Original attribution and rights notices remain in the PDF and page renders.\n")
    return {"id": paper["id"], "pages": pages, "embedded_files": record["embedded_files"]}


if __name__ == "__main__":
    selected = set(sys.argv[1:])
    known = {p["id"] for p in SOURCES["papers"]}
    if selected - known:
        raise SystemExit(f"Unknown paper IDs: {sorted(selected - known)}")
    papers = [p for p in SOURCES["papers"] if not selected or p["id"] in selected]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        for result in executor.map(extract, papers):
            print(json.dumps(result), flush=True)
