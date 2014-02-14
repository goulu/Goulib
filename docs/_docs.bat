call make clean 
python generate_modules.py -t -s rst -d .\modules ..\Goulib
call make html
start _build\html\index.html
pause
cd ..