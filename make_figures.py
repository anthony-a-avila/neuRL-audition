"""Regenerate the notebook's two figures as PNG files.

All of the numerics live in td_dopamine.ipynb. Rather than duplicating them here,
this script executes that notebook's code cells in order and saves whatever
figures they produce, so the PNGs cannot drift away from the notebook.

    python make_figures.py                  # writes figures/*.png at 200 dpi
    python make_figures.py --dpi 300        # higher resolution
    python make_figures.py --outdir /tmp    # somewhere else
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must precede the notebook's own pyplot import

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "td_dopamine.ipynb"

# In the order the notebook creates them.
FIGURE_NAMES = [
    "fig1_three_cases.png",
    "fig2_spike_rasters.png",
]


def run_notebook_code(notebook):
    """Execute every code cell of a notebook in one shared namespace."""
    cells = json.loads(notebook.read_text())["cells"]
    namespace = {"__name__": "__notebook__", "__file__": str(notebook)}
    for index, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if not source.strip():
            continue
        code = compile(source, f"{notebook.name}[cell {index}]", "exec")
        exec(code, namespace)
    return namespace


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--outdir", type=Path, default=HERE / "figures")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    if not NOTEBOOK.exists():
        raise SystemExit(f"notebook not found: {NOTEBOOK}")

    with warnings.catch_warnings():
        # The cells end in plt.show(), which is a no-op under Agg and says so.
        # The figures are still registered with pyplot afterwards.
        warnings.filterwarnings("ignore", message=".*non-interactive.*")
        run_notebook_code(NOTEBOOK)

    figures = [plt.figure(num) for num in plt.get_fignums()]
    if len(figures) != len(FIGURE_NAMES):
        raise SystemExit(
            f"expected {len(FIGURE_NAMES)} figures, notebook produced {len(figures)}; "
            "update FIGURE_NAMES if the notebook gained or lost a figure"
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    for figure, name in zip(figures, FIGURE_NAMES):
        path = args.outdir / name
        figure.savefig(path, dpi=args.dpi)
        print(f"wrote {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}")


if __name__ == "__main__":
    main()
