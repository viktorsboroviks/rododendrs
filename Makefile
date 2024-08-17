.PHONY: \
	format \
	clang-format \
	lint \
	cppcheck

format: clang-format

clang-format: include/rododendrs.hpp
	clang-format -i $^

lint: cppcheck

cppcheck: \
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
