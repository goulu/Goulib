cd goulib
rem for %%1 in (*.py) do pythoscope -q -t nose %1
cd ..
nosetests --with-coverage --cover-package=Goulib
call activate.bat pypy
nosetests
call activate.bat py34
nosetests
call deactivate.bat