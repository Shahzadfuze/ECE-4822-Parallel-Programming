all: x.exe

x.exe: x.o
	nvcc -allow-unsupported-compiler -g -lm x.o -o x.exe

x.o: x.cu Makefile
	nvcc -allow-unsupported-compiler -c x.cu -o x.o

clean:
	rm -f x.o x.exe
