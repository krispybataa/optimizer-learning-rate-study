# Define the path to the LaTeX directory
$latexDir = "docs/latex"

Write-Host "Cleaning auxiliary files..."
# Use -cd to tell latexmk to change to the directory of the target file
latexmk -C -cd "$latexDir/main.tex"

Write-Host "Building PDF..."
# -pdf generates the pdf, -f forces build, -cd handles relative paths for includes
latexmk -pdf -f -cd "$latexDir/main.tex"

Write-Host "Done."