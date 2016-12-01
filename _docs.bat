cls -C
cd docs
call sphinx-apidoc ..\Goulib -eo modules 
del modules\Goulib.rst
del _build\html\index.html
call make html
start _build\html\index.html
cd ..