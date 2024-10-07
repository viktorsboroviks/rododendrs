.PHONY: \
	all \
	benchmarks \
	format \
	format-cpp \
	lint \
	lint-cpp \
	clean \
	distclean

all: benchmarks

venv: requirements.txt
	python3 -m venv venv
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

format: format-cpp

format-cpp: \
		include/rododendrs.hpp \
		benchmarks/sampling_f/sampling_f.cpp
	clang-format -i $^

lint: lint-cpp

lint-cpp: \
		include/rododendrs.hpp \
		benchmarks/sampling_f/sampling_f.cpp
	cppcheck \
		--enable=warning,portability,performance \
		--enable=style,information \
		--enable=missingInclude \
		--inconclusive \
		--library=std,posix,gnu \
		--platform=unix64 \
		--language=c++ \
		--std=c++20 \
		--inline-suppr \
		--check-level=exhaustive \
		--suppress=missingIncludeSystem \
		--suppress=checkersReport \
		--checkers-report=cppcheck_report.txt \
		-I./include \
		$^

clean:
	rm -rf *.o
	rm -rf *.csv
	rm -rf *.html

distclean: clean
	rm -rf venv
	rm -rf vasarniica
