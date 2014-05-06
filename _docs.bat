cd goulib
sphinx-apidoc . -eo ..\docs\modules 
rem del ..\docs\modules\Goulib.rst
cd ..
cd docs
call make html
start _build\html\index.html
cd ..