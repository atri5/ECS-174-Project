report:
	pdflatex -synctex=1 -interaction=nonstopmode --shell-escape report.tex
	bibtex report
	pdflatex -synctex=1 -interaction=nonstopmode --shell-escape report.tex
	pdflatex -synctex=1 -interaction=nonstopmode --shell-escape report.tex

clean:
	rm -f *.aux *.fdb_latexmk *.fls *.log *.out *.pyg *.synctex.gz *.bbl *.blg *.synctex*