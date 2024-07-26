.PHONY: format clang-format

format: clang-format

clang-format: include/rododendrs.hpp
	clang-format -i $^
