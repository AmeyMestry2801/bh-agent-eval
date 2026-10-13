"""Convert a '# %%' percent-format .py file into a .ipynb (no outputs).

Usage (from project root):  python notebooks/_py2nb.py notebooks/01_eda.py
"""
import sys
from pathlib import Path

import nbformat as nbf


def convert(src: Path) -> Path:
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        body = "\n".join(buf).strip("\n")
        if not body:
            return
        if kind == "markdown":
            text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.split("\n"))
            cells.append(nbf.v4.new_markdown_cell(text))
        else:
            cells.append(nbf.v4.new_code_cell(body))

    for line in src.read_text(encoding="utf-8").split("\n"):
        if line.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in line else "code"
            buf = []
        else:
            buf.append(line)
    flush()

    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    out = src.with_suffix(".ipynb")
    nbf.write(nb, out)
    return out


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print("wrote", convert(Path(arg)))
