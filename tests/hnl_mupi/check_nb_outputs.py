"""Pre-commit check: the staged version of each notebook has no outputs or execution counts.

Reads the index blob (``git show :path``), so it sees what will be committed whether or not the
local nbstripout git filter ran; falls back to the file for paths not in the index.
usage: check_nb_outputs.py <notebook.ipynb> ...
"""
import subprocess
import sys

import nbformat


def staged_text(path):
    r = subprocess.run(["git", "show", f":{path}"], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else open(path).read()


def dirty_cells(path):
    nb = nbformat.reads(staged_text(path), as_version=4)
    return [i for i, c in enumerate(nb.cells)
            if c.cell_type == "code" and (c.get("outputs") or c.get("execution_count") is not None)]


if __name__ == "__main__":
    bad = {p: dirty_cells(p) for p in sys.argv[1:]}
    bad = {p: cells for p, cells in bad.items() if cells}
    for p, cells in bad.items():
        print(f"{p}: outputs/execution counts in cells {cells}; strip them (nbstripout {p}) and re-add")
    sys.exit(1 if bad else 0)
