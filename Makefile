# Compiler and flags
CXX = mpic++
CXXFLAGS = -O3 -std=c++17 -mavx2 -mfma
INSTR_CXX = scorep-mpicxx
INSTR_CXX_ONLY = scorep --nocompiler mpicxx
INSTR_CXX_FLAGS = -O3 -std=c++17 -mavx2 -mfma -g

# Source files
SRCS = main.cpp raytracer.cpp scene.cpp
OBJS = $(SRCS:.cpp=.o)

# Target executable
TARGET = snowman

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

instr:
	$(INSTR_CXX) $(INSTR_CXX_FLAGS)  $(SRCS) -o snowman-instr

instr-only:
	$(INSTR_CXX_ONLY) $(INSTR_CXX_FLAGS)  $(SRCS) -o snowman-instr-only
