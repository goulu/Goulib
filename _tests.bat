cd goulib
for %%1 in (*.py) do pythoscope -q -t nose %1
cd ..
nosetests --with-coverage --cover-package=Goulib
