from pathlib import Path
import fitz

pdf_dir = Path("./hallucination")
md_dir = Path("./mds")
md_dir.mkdir(exist_ok=True)

for pdf_file in pdf_dir.glob("*.pdf"):
    doc = fitz.open(pdf_file)
    md_file = md_dir / (pdf_file.stem + ".md")
    with open(md_file, "w", encoding="utf-8") as f:
        for page in doc:
            f.write(page.get_text() + "\n")
