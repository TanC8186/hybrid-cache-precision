.PHONY: all clean

all: main.pdf

main.pdf: main.tex main.bib selector_audit_table.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	latexmk -C main.tex
