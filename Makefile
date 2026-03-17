.DEFAULT_GOAL := help

PYTHON ?= python3

.PHONY: help install cpp validators subjects quick regex single double triple truncation all ddmax ddmax-subjects

help:
	@printf '%s\n' \
		'Common targets:' \
		'  make install         Install Python dependencies from requirements.txt' \
		'  make cpp             Build the C++ betaMax backend' \
		'  make validators      Build native regex validators (optional)' \
		'  make subjects        Build grammar/subject artifacts for JSON/INI/DOT/... benchmarks' \
		'  make quick           Run a small smoke benchmark' \
		'  make regex           Run the main regex benchmark suite (single/double/triple)' \
		'  make single          Run single-mutation regex benchmarks' \
		'  make double          Run double-mutation regex benchmarks' \
		'  make triple          Run triple-mutation regex benchmarks' \
		'  make truncation      Run the JSON truncation benchmark (requires subject binaries)' \
		'  make all             Run regex + truncation benchmarks' \
		'  make ddmax           Run DDMax regex benchmarks' \
		'  make ddmax-subjects  Run DDMax subject benchmarks (requires erepair.jar + subject DBs)' \
		'' \
		'Useful env vars:' \
		'  PYTHON=<python-bin>  Choose the Python interpreter (default: python3)' \
		'  MAX_WORKERS=<n>      Parallel workers forwarded to bm_*.py' \
		'  DB_PREFIX=<name>     Prefix output DB names, e.g. DB_PREFIX=trial make regex'

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

cpp:
	cmake -S betamax_cpp -B betamax_cpp/build -DCMAKE_BUILD_TYPE=Release
	cmake --build betamax_cpp/build -j

validators:
	./validators/build_validators.sh

subjects:
	./build_all.sh

quick:
	./run_bms.sh quick

regex:
	./run_bms.sh regex

single:
	./run_bms.sh single

double:
	./run_bms.sh double

triple:
	./run_bms.sh triple

truncation:
	./run_bms.sh truncation

all:
	./run_bms.sh all

ddmax:
	./run_ddmax.sh regex

ddmax-subjects:
	./run_ddmax.sh subjects
