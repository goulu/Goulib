rem sphinx-apidoc -f -F -H "Goulib" -A "Ph. Guglielmetti, https://github.com/goulu/Goulib" -o . ../Goulib
call make clean 
rem ATTENTION : enlever les rem ci-dessus resette les options de Sphinx dans conf.py ainsi que les modifs des fichiers .rst
python generate_modules.py -t -s rst -d .\modules ..\Goulib
call make html
start _build\html\index.html
pause