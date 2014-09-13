for %%1 in (Goulib/*.py) do pythoscope -q -t nose %1
nosetests --with-coverage --cover-package=Goulib
call activate.bat py34
nosetests
call deactivate.bat