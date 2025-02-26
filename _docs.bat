cls -C
call pip install sphinx sphinx_rtd_theme
cd docs
call sphinx-apidoc ..\goulib -eo modules 
del modules\goulib.rst
del build\html\index.html
call make html
start build\html\index.html
cd ..