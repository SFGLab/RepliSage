# Developer notes

Short checklists for maintainers. Run everything from the repository root.

## Layout

| Path | What it holds |
|---|---|
| `src/` | the code, installed as the package `RepliSage` (PyPI name `pyRepliSage`) |
| `src/_version.py` | the only place with the version (`setup.py` reads it) |
| `markdown/` | documentation source (MkDocs, theme Material); `mkdocs.yml` holds the navigation |
| `docs/tikz/` | TikZ sources of the biology figures; the SVGs live in `markdown/assets/diagrams/` |
| `demo/` | quick checks that need no simulation: `test_config.py`, `test_inputs.py`, `test_normalisation.py` |
| `tests/` | pytest tests |
| `.github/workflows/` | `docs.yml` (docs deploy) and `python-publish.yml` (PyPI upload) |

## Everyday checks

```bash
pip install -e ".[viz,test,docs]"
python run_tests.py          # demo checks + fast tests, one summary table (about 1-2 min)
python run_tests.py --all    # also the slow tests (OpenMM and full pipelines)
pytest                       # only the fast pytest tests (pytest.ini skips the slow ones)
pytest -m slow               # only the slow tests
pytest tests/test_video_bar.py -k bar   # one file or test
```

`run_tests.py` runs `demo/test_config.py`, `demo/test_inputs.py` and `demo/test_normalisation.py` (no simulation needed), then pytest, and ends with one PASS/FAIL line per step plus the tail of any failing output. Test regions are small (4 Mb, 100 beads = 40 kb per bead) because beads must be between 200 bp and 50 kb.

## Documentation

```bash
pip install -r requirements-docs.txt   # first time only
mkdocs serve                           # local preview at http://127.0.0.1:8000
mkdocs build --strict --clean          # must finish without warnings (broken links, pages missing from the nav)
```

* A new page must be added to `nav` in `mkdocs.yml`, otherwise it is not shown.
* Defaults written in the docs (`config.md`, `cli.md`, `mc_model.md`, `replication_model.md`) must match `src/config.ini` and `src/config/arg_defs.py`. After changing a default, search the docs for the old value: `grep -rn "<old value>" markdown`.
* Rebuild a TikZ figure: `pdflatex fig.tex && pdftocairo -svg fig.pdf fig.svg`, then copy the SVG to `markdown/assets/diagrams/`.
* The site is deployed by `.github/workflows/docs.yml` on every release, on pushes to `main` that touch `markdown/` or `mkdocs.yml`, and on demand (Actions tab). One-time setting: Settings → Pages → Source = branch `gh-pages`, root.

## Release checklist

1. Set the version in `src/_version.py` (for example `__version__ = "1.0.0"`).
2. Run the everyday checks and `mkdocs build --strict --clean`.
3. Check that the package builds:
   ```bash
   rm -rf build dist *.egg-info
   pip install build twine
   python -m build
   twine check dist/*
   ```
4. Commit and push (this also redeploys the docs):
   ```bash
   git status
   git add -A
   git commit -m "v1.0.0: <short summary>"
   git push origin main
   ```
5. Tag and publish the release:
   ```bash
   git tag -a v1.0.0 -m "RepliSage 1.0.0"
   git push origin v1.0.0
   gh release create v1.0.0 --title "RepliSage 1.0.0" --generate-notes
   ```
   Publishing the release starts two workflows: `python-publish.yml` uploads to PyPI (needs the repository secret `PYPI_API_TOKEN`) and `docs.yml` redeploys the site.
6. Check the result in a clean environment and in the browser:
   ```bash
   pip install --upgrade pyRepliSage==1.0.0
   replisage --help
   ```
   Open https://sfglab.github.io/RepliSage/ and look at the new pages. If the site returns 404, check Settings → Pages.

## Notes

* Run logging levels: no flag = stages, progress bars and results; `-v` = input and output tables; `-vv` = detailed info and library warnings; `-vvv` = everything (VTK/OpenGL messages included).
* Movies and the HTML viewer share one palette (`src/viz3d/palette.py`); change colours there only.
