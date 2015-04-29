Optimal Inference After Model Selection
==========

The main files in the top-level directory are `paper.tex` (the LaTeX source), `biblio.bib` (bibliography), and `mycommands.sty` (LaTeX style file).

Figures directory
------

Figures appearing in paper are in `figs/`.

Code directory
------

The `code/` directory contains source code for generating figures and running simulations. Each figure in `paper.tex` is accompanied by a comment giving a path to the file that generated it. There are three subdirectories:

- `code/lasso_example/`: Source code for the simulation in Section 7.

- `code/twodim_example/`: Source code for generating the expected confidence interval lengths for data splitting and data carving in Figure 4b.

- `code/misc_plots/`: Source code for generating other plots interspersed throughout the paper.
