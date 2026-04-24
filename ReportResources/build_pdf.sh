#!/bin/bash
# Move to the report directory
cd /home/xux1/Northeastern/CS6120/RAG/ReportResources/

echo "Checking for LaTeX compilers..."

# Check if pdflatex is installed
if command -v pdflatex &> /dev/null; then
    echo "Found pdflatex. Compiling..."
    # Run twice to resolve references/formatting
    pdflatex -interaction=nonstopmode report.tex
    pdflatex -interaction=nonstopmode report.tex
else
    echo "pdflatex not found on this system."
    echo "Since this might be an HPC environment without sudo, downloading Tectonic (a self-contained LaTeX engine)..."
    
    # Download Tectonic (standalone LaTeX compiler that pulls dependencies on the fly)
    curl --proto '=https' --tlsv1.2 -fsSL https://drop-sh.fullyjustified.net | sh
    
    echo "Compiling with Tectonic..."
    ./tectonic report.tex
fi

if [ -f "report.pdf" ]; then
    echo "✅ Successfully generated report.pdf!"
else
    echo "❌ Failed to generate the PDF."
fi