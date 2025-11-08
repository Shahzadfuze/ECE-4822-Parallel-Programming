# file:
#
# Note that this make file uses "-g" to compile and link "debug".
#

# define a target for the application
#
all: x.exe

# define a target to link the application
#
x.exe: x.o x_00.o
	g++ -g -O2 -fopenmp -lm x.o x_00.o -o x.exe

# define targets to compile the source code
#
x.o: x.cc x.h Makefile
	g++ -g -O2 -fopenmp -c x.cc -o x.o

x_00.o: x_00.cc x.h Makefile
	g++ -g -O2 -fopenmp -c x_00.cc -o x_00.o

# define a target to clean the directory
#
clean:
	rm -f x.o x_00.o x.exe

#
# end of file
