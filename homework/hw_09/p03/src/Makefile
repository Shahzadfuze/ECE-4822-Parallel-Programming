# file: example_vector_add/Makefile
#

# define a target for the application
#
all: fuze.exe

# define a target to compile and link the application
#
fuze.exe: Makefile main.cu functs.cu kernels.cu header.cuh
	nvcc -allow-unsupported-compiler --compiler-options -fopenmp -o add.exe main.cu functs.cu kernels.cu

# define a target to clean the directory
#
clean:
	rm -f add.exe

#
# end of file
