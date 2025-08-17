.PHONY: \
	all \
	benchmarks \
	tests \
	format \
	format-cpp \
	lint \
	lint-cpp \
	clean \
	distclean

all: benchmarks tests

PYTHON_VERSION := python3.12
venv: requirements.txt
	$(PYTHON_VERSION) -m venv venv
	./venv/bin/pip3 install --no-cache-dir --requirement requirements.txt

vasarniica:
	git clone git@github.com:viktorsboroviks/vasarniica.git
	cd vasarniica; git checkout v1.5

benchmarks: sampling_f.csv venv vasarniica
	PYTHONPATH=${PYTHONPATH}:vasarniica/python ./venv/bin/python3 \
		benchmarks/sampling_f/plot_sampling_f.py \
		--input-csv sampling_f.csv \
		--output-file result.html

sampling_f.o: \
		benchmarks/sampling_f/sampling_f.cpp
	g++ -Wall -Wextra -Werror -Wpedantic \
		-std=c++20 -O3 \
		-I./include \
		benchmarks/sampling_f/sampling_f.cpp -o $@

sampling_f.csv: sampling_f.o
	./sampling_f.o > $@

tests: \
		./tests/test_cdf.py \
		test_rododendrs.o \
		test_cdf.o
	./venv/bin/python3 ./tests/test_cdf.py
	./test_rododendrs.o

test_rododendrs.o: tests/test_rododendrs.cpp
	g++ -Wall -Wextra -Werror -Wpedantic \
		-std=c++20 -O3 \
		-I./include \
		tests/test_rododendrs.cpp -o $@

test_cdf.o: tests/test_cdf.cpp
	g++ -Wall -Wextra -Werror -Wpedantic \
		-std=c++20 -O3 \
		-I./include \
		tests/test_cdf.cpp -o $@

format: format-cpp

format-cpp: \
		include/rododendrs.hpp \
		benchmarks/sampling_f/sampling_f.cpp \
		tests/test_rododendrs.cpp \
		tests/test_cdf.cpp
	clang-format -i $^

lint: lint-cpp

lint-cpp: \
		include/rododendrs.hpp \
		benchmarks/sampling_f/sampling_f.cpp \
		tests/test_rododendrs.cpp \
		tests/test_cdf.cpp
	cppcheck \
		--enable=warning,portability,performance \
		--enable=style,information \
		--inconclusive \
		--library=std,posix,gnu \
		--platform=unix64 \
		--language=c++ \
		--std=c++20 \
		--inline-suppr \
		--check-level=exhaustive \
		--suppress=checkersReport \
		--checkers-report=cppcheck_report.txt \
		$^

clean:
	rm -rf *.o
	rm -rf *.csv
	rm -rf *.html

distclean: clean
	rm -rf venv
	rm -rf vasarniica
