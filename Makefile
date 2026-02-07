# Compiler and Assembler
CXX = g++
AS = as

# Directories
CPP_DIR = src/cpp
ASM_DIR = src/asm
OBJ_DIR = build

# Flags
CXXFLAGS = -O2 -std=c++17
ASFLAGS = -g

# Source Files
CPP_SRCS = $(wildcard $(CPP_DIR)/*.cpp)
ASM_SRCS = $(wildcard $(ASM_DIR)/*.s)

# Object Files (place them in a build directory)
CPP_OBJS = $(patsubst $(CPP_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(CPP_SRCS))
ASM_OBJS = $(patsubst $(ASM_DIR)/%.s, $(OBJ_DIR)/%.o, $(ASM_SRCS))
OBJS = $(CPP_OBJS) $(ASM_OBJS)

# Target: benchmark
benchmark: $(OBJ_DIR) $(OBJS)
	$(CXX) $(CXXFLAGS) -o benchmark $(OBJS)

# Rule to create build directory
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Rule for C++ files
$(OBJ_DIR)/%.o: $(CPP_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule for Assembly files
$(OBJ_DIR)/%.o: $(ASM_DIR)/%.s
	$(AS) $(ASFLAGS) -c $< -o $@

# Clean up
clean:
	rm -rf $(OBJ_DIR) benchmark

.PHONY: clean benchmark