cd goulib
rem for %%i in (*.py) do pythoscope -q -t nose %%i
cd ..
nosetests --with-coverage --cover-package=Goulib
