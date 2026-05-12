# LaTeX Compilation Handoff

## Goal

Produce a compilable pdfLaTeX document at `docs/latex/main.tex` that generates a complete
PDF of the Optimizer and Learning Rate Study paper (66 pages) with:
- All `\ref{}` cross-references resolved (no `??`)
- All 84-run result figures included or placeholdered
- All source code appendices typeset from clean ASCII Python files
- No fatal LaTeX errors on `latexmk -pdf main.tex`

---

## Current State

**The document almost compiles.** The first `latexmk` run produced a 66-page PDF but
aborted with a fatal UTF-8 error inside `visualize_clean.py`, which prevented the `.aux`
file from being fully written. As a result:

- 29 `\ref{}` calls show as `??` (undefined) — caused by the crash, not by missing labels
- The second `latexmk` run said "Nothing to do" — latexmk cached the error state and
  will not retry without being forced with `-f`
- The longtable "Column widths have changed" warning is non-fatal and resolves on the
  second clean pass

**Root cause of crash:** `visualize_clean.py` was created with three `—` (U+2014 EM DASH)
characters still present in inline comments. These were missed when only the `set_title()`
string literals were cleaned. pdflatex cannot handle them because `—` is not in the
`literate` substitution list for the `python` lstlisting style.

---

## Files in Flight

| File | Status | Notes |
|---|---|---|
| `docs/latex/main.tex` | Done | `\setlength{\headheight}{14pt}` added; preamble complete |
| `docs/latex/latexmkrc` | Done | `$pdf_mode=1`, `$max_repeat=5` |
| `docs/latex/sections/02_dataset.tex` | Done | All labels present and correctly placed |
| `docs/latex/sections/03_preprocessing.tex` | Done | All labels present and correctly placed |
| `docs/latex/sections/04_pipeline.tex` | Done | TikZ fixed; named colors defined |
| `docs/latex/sections/05_methodology.tex` | Done | longtable wrapped with `\setstretch{1}` + `\extrarowheight{0pt}` |
| `docs/latex/sections/06_results.tex` | Done | longtable wrapped; heatmap figures placeholdered |
| `docs/latex/sections/07_discussion.tex` | Done | |
| `docs/latex/sections/08_conclusion.tex` | Done | |
| `docs/latex/sections/appx-acc-loss.tex` | Done | 84 subfigures; all 168 PNGs verified present |
| `docs/latex/sections/appx-auc-roc.tex` | Done | Same structure |
| `docs/latex/sections/appx-confusion.tex` | Done | Same structure |
| `docs/latex/sections/appx-source-code.tex` | Done | Points to `_clean.py` variants |
| `src/run_all_clean.py` | Done | `═` `—` `⚠️` replaced |
| `src/train_clean.py` | Done | `—` replaced in all step-comments and print string |
| `src/preprocessing_clean.py` | Done | `⚠` `∩` `—` `✓` `✗` `…` replaced |
| `src/visualize_clean.py` | **BROKEN** | Still contains `—` on lines 11, 20, 21 — see Next Step |
| `src/evaluate.py` | Done | No problematic Unicode; used directly |
| `src/model_builder.py` | Done | No problematic Unicode; used directly |

---

## What's Changed (this session)

1. **Preamble** (`main.tex`): Added `lmodern`, `[T1]{fontenc}`, `textcomp`, `microtype`;
   upgraded xcolor to `[dvipsnames,svgnames,x11names,table,HTML]`; removed standalone
   `colortbl`; changed `basicstyle=\tiny` to `\scriptsize\ttfamily`; added
   `\setlength{\headheight}{14pt}`
2. **Title page** (`main.tex`): Removed tabular environment; removed `\and`; simple
   centered layout with `\quad` between author names
3. **TikZ pipeline** (`04_pipeline.tex`): Eliminated parameterized `rndbox` style that
   caused `#1` to be misread as HTML hex color; defined 8 named colors with `\definecolor`
4. **Em-dash sweep**: Replaced all `---` and U+2014 `—` characters document-wide with
   `:`, `,`, or `-` depending on context
5. **Longtable fix** (`05_methodology.tex`, `06_results.tex`): Wrapped both longtables
   with `\begingroup\setstretch{1}\setlength{\extrarowheight}{0pt}...\endgroup`
6. **Heatmap figures** (`06_results.tex`): Wrapped in `\fbox{\parbox{...}}` placeholders
   since notebook has not been run yet
7. **Notebook** (`notebooks/03_results_analysis.ipynb`): Added `os.makedirs` and
   `plt.savefig()` calls before `plt.show()` for both heatmap cells
8. **Clean source files**: Created `_clean.py` variants for 4 of 6 src files;
   `appx-source-code.tex` updated to reference them
9. **latexmkrc**: Created with `$pdf_mode=1`, `$pdflatex=...`, `$max_repeat=5`
10. **Label audit**: Confirmed all 26 required `\label{}` commands are present and
    correctly placed inside their float environments

---

## Failed Attempts

| Attempt | What went wrong |
|---|---|
| `\begin{tabular}` on title page | Fatal error: tabular not closed before `\end{titlepage}` |
| `colortbl` standalone + xcolor `table` option | Conflict; fixed by removing `colortbl`, keeping xcolor `table` |
| `\tiny\ttfamily` in lstdefinestyle | pdfTeX font expansion error; fixed with `\scriptsize` |
| `rndbox/.style={draw=#1!80!black}` TikZ style | `#1` parsed as start of `#RRGGBB` hex color, producing `#180` error |
| Longtable inside `\onehalfspacing` without `\setstretch{1}` | "Infinite glue shrinkage" on page breaks |
| Creating `visualize_clean.py` | Only replaced `—` in `set_title()` calls; missed 3 inline comment occurrences on lines 11, 20, 21 |

---

## Next Step

### Step 1 — Fix `src/visualize_clean.py` (ONE fix, three lines)

Open `src/visualize_clean.py` and replace the three remaining `—` (U+2014) characters:

**Line 11** — before:
```python
matplotlib.use("Agg")  # non-interactive backend — safe for headless runs
```
after:
```python
matplotlib.use("Agg")  # non-interactive backend - safe for headless runs
```

**Line 20** — before:
```python
_CANCER_COLOR = "#C0392B"   # red — cancer class
```
after:
```python
_CANCER_COLOR = "#C0392B"   # red - cancer class
```

**Line 21** — before:
```python
_NORMAL_COLOR = "#2980B9"   # blue — no_cancer class
```
after:
```python
_NORMAL_COLOR = "#2980B9"   # blue - no_cancer class
```

### Step 2 — Clean stale build artifacts

From `docs/latex/`, delete the following files so latexmk starts fresh:

```
main.aux  main.toc  main.lof  main.lot  main.out
main.fls  main.synctex.gz  main.pdf
```

On Windows PowerShell:
```powershell
cd docs\latex
Remove-Item main.aux, main.toc, main.lof, main.lot, main.out, main.fls, main.synctex.gz, main.pdf -ErrorAction SilentlyContinue
```

### Step 3 — Recompile

```powershell
latexmk -pdf main.tex
```

latexmk will run up to 5 passes (per `$max_repeat=5` in latexmkrc). By pass 2:
- longtable column-width warnings will resolve
- All `\ref{}` calls will resolve (labels are correctly placed; the `??` were caused
  entirely by the UTF-8 crash aborting `.aux` file write)
- The remaining warnings (`main.out changed`, `rerun for outlines`) will resolve by
  pass 3 at latest

### Expected output after fix

A clean PDF with no fatal errors. Remaining non-fatal warnings that will persist
(do not block compilation):
- `longtable: Table widths have changed` — resolves after pass 2
- `Package rerunfilecheck: File main.out has changed` — resolves after pass 3
- Heatmap figures show as `[Figure pending: ...]` boxes — intentional placeholder until
  `notebooks/03_results_analysis.ipynb` is run
