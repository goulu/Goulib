rem sphinx-apidoc -f -F -H "Goulib" -A "Ph. Guglielmetti, https://github.com/goulu/Goulib" -o . .
rem call make clean ATTENTION DANGEREUX !
call make html
move _build/html/*.* .
start index.html
pause