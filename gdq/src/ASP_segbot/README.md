# ASP-code
reasoning and planning in MDP Domain.

## asp_files
asp_files provides some codes which plan in MDP.

## asp2py
this provides a parser which converts output from clingo to strings.

## original_asp_files
it is for just in case. This directory is the original code when I get this code

# installation
You need to install Clingo on Ubuntu 18.04. 
To install Clingo, follow the instruction.

## requirement 
- a c++14 conforming compiler
	- at least gcc version 4.9
	- clang version 3.1 (using either libstdc++ provided by gcc 4.9 or libc++)
	- at least msvc++ 14.0 (Visual Studio 2015 Update 3)
	- other compilers might work
- the cmake build system
	- at least version 3.3 is recommended
	- at least version 3.1 is required
## dependences
- the bison parser generator
	- at least version 2.5
	- version 3.0 produces harmless warnings (to stay backwards-compatible)
- the re2c lexer generator
	- at least version 0.13
	- version 0.13.5 is used for development

## build, install
	
	git clone https://github.com/potassco/clingo.git <path to clingo>
	cd <path to clingo>
	git submodule update --init --recursive
	
If you haven't install lua, bison and re2c yet;

	sudo apt install lua5.3 liblua5.3-dev 
	sudo apt install bison
	sudo apt install re2c

	make
	
	make install

To test the ASP files, run this command: 

	clingo -c n=10 *.asp -n 0

where n-1 is the maximum number of actions allowed in planning. 
where -n means to comute at most <n> models (0 for all)
