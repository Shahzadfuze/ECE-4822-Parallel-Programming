CXX = g++
CXXFLAGS = -O2 -std=c++17 -I/data/isip/tools/linux_x64/common/anaconda3/conda/envs/nedc_03.11/include/eigen3
EXEC = p01.exe
SRC = p01.cc
OBJ = p01.o
DEPS = $(NEDC_NFC)/lib/libdsp.a
INCLUDES = $(ISIP_INCLUDE)

all: $(EXEC)

$(EXEC): $(OBJ) $(DEPS)
	$(CXX) $(CXXFLAGS) -o $(EXEC) $(OBJ) $(ISIP_LIBS)

$(OBJ): $(SRC) $(DEPS)
	$(CXX) $(CXXFLAGS) -c $(SRC) $(INCLUDES) -o $(OBJ)

clean:
	rm -f $(EXEC) $(OBJ)
