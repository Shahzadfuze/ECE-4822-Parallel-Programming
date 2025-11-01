# file:
#
# Note that this make file uses "-g" to compile and link "debug".
#

# define a target for the application
#
all: main.exe

# define a target to link the application
#
main.exe: main.o
	g++ -g -lm main.o  -o main.exe

# define targets to compile the source code
#
main.o: main.cc example.h Makefile
	g++ -g -c main.cc -o main.o

# define a target to clean the directory
#
clean:
	rm -f main.o main.exe

#
# end of file
