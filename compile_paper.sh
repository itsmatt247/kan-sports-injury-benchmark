#!/bin/bash

# Stop on error
set -e

echo "🚀 Compiling TMLR Paper..."

# Change into the paper directory to ensure all relative paths (bibtex, figures) work
cd paper

# 1. First Pass
pdflatex -interaction=nonstopmode main.tex

# 2. Bibliography (Now finds main.bib correctly because we are in the same folder)
bibtex main

# 3. Second Pass
pdflatex -interaction=nonstopmode main.tex

# 4. Final Pass
pdflatex -interaction=nonstopmode main.tex

# Return to root
cd ..

echo "✅ Compilation Complete!"
echo "📄 PDF is located at: paper/main.pdf"

#./compile_paper.sh (run this to compile the paper)