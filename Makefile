.PHONY: \
	all \
	benchmarks \
	format \
	format-cpp \
	lint \
	lint-cpp

all: benchmarks

benchmarks: sampling_f.o

sampling_f.o: \
		benchmarks/sampling_f/sampling_f.cpp
	g++ -Wall -Wextra -Werror -Wpedantic \
		-std=c++20 -O3 \
		-I./include \
		benchmarks/sampling_f/sampling_f.cpp -o $@

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
	rm -rfv *.txt
