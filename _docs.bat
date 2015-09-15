cls -C
cd goulib
call sphinx-apidoc . -eo ..\docs\modules 
cd ..
cd docs
del modules\Goulib.rst
del _build\html\index.html
call make html
start _build\html\index.html
cd ..