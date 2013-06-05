rem sphinx-apidoc -f -F -H "Goulib" -A "Ph. Guglielmetti, https://github.com/goulu/Goulib" -o . ../Goulib
rem call make clean ATTENTION DANGEREUX !
python generate_modules.py -t -s rst -d .\modules ..\Goulib
call make html
start _build\html\index.html
pause