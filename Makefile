.PHONY: \
	format \
	format-cpp \
	lint \
	lint-cpp

format: format-cpp

format-cpp: include/rododendrs.hpp
	clang-format -i $^

lint: lint-cpp

lint-cpp: \
		include/rododendrs.hpp
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
