cls -C
cd docs
call sphinx-apidoc ..\goulib -eo modules 
del modules\goulib.rst
del _build\html\index.html
call make html
start _build\html\index.html
cd ..