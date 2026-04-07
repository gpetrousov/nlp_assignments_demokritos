TEX_IMAGE=xelatex
TEX_IMAGE_TAG=full

all:
	docker run --rm -v $(PWD):$(PWD) -w $(PWD) $(TEX_IMAGE):$(TEX_IMAGE_TAG) xelatex report.tex

bib:
	docker run --rm -v $(PWD):$(PWD) -w $(PWD) $(TEX_IMAGE):$(TEX_IMAGE_TAG) bibtex report

clean:
	rm -f report.{aux,bbl,blg,log,out,toc,lot,lof}

full: clean all bib
	docker run --rm -v $(PWD):$(PWD) -w $(PWD) $(TEX_IMAGE):$(TEX_IMAGE_TAG) xelatex report.tex
	docker run --rm -v $(PWD):$(PWD) -w $(PWD) $(TEX_IMAGE):$(TEX_IMAGE_TAG) xelatex report.tex

read:
	epdfview report.pdf
