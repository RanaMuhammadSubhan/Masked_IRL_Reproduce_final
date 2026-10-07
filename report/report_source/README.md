# Technical report source

make_docx.py reads the canonical CSVs and unchanged PNG figures from a --package directory. It creates the technical DOCX and an expected static TOC index; it never trains or changes results. Requires python-docx. Example from package root:

`python report/report_source/make_docx.py --package . --output work/report/Masked_IRL_Reproduction_Implementation_and_Learnings.docx`

The delivered document was rendered through the documents skill render_docx.py, task-local LibreOffice 26.2.6.3 and Poppler, then all 23 pages were visually inspected. It contains 21 pages including front matter before two appendix pages, 15 tables, four unchanged figures, native Word equations and a populated linked static TOC. After any editing, regenerate/recheck the static TOC page numbers; it is not a live Word field. The professor PDF is copied unchanged and has separate existing QA evidence.
