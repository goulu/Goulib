set/p pwd="BOBST proxy password for %username% ? "
set http_proxy=http://%username%:%pwd%@80.254.148.58:8080
set pwd=

pip install -r requirements.txt

pip install Sphinx