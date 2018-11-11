(TeX-add-style-hook
 "solution"
 (lambda ()
   (TeX-run-style-hooks
    "latex2e"
    "standalone"
    "standalone10"
    "amsmath"
    "sourcesanspro"
    "tikz")
   (TeX-add-symbols
    "mathup"
    "midarrow"))
 :latex)

