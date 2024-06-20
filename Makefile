# Define directories
BASE_DIR := .
NLA_DIR := ./NLA
OBJ_DIR := ./obj

# Collect all .cpp files in the base directory and NLA directory
SRC_FILES := $(wildcard $(BASE_DIR)/*.cpp) $(wildcard $(NLA_DIR)/*.cpp)

# Define the target executable
TARGET := fingerprint_optimizer

# Define the compiler and compiler flags
CXX := g++
CXXFLAGS := -std=c++17 -O2 -shared-libgcc -MMD -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
LDFLAGS := -Xpreprocessor -fopenmp -L/opt/homebrew/opt/libomp/lib -lomp

# Define the object files (with paths relative to OBJ_DIR)
OBJ_FILES := $(patsubst $(BASE_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(wildcard $(BASE_DIR)/*.cpp)) \
             $(patsubst $(NLA_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(wildcard $(NLA_DIR)/*.cpp))

# Define the dependency files
DEP_FILES := $(OBJ_FILES:.o=.d)

# Rule to create the obj directory if it doesn't exist
$(shell mkdir -p $(OBJ_DIR))

# Rule to build the target executable
$(TARGET): $(OBJ_FILES)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

# Rule to build object files
$(OBJ_DIR)/%.o: $(BASE_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(NLA_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Include dependency files
-include $(DEP_FILES)

# Clean target to remove generated files
clean:
	rm -rf $(OBJ_DIR) $(TARGET)

# Phony targets
.PHONY: all clean

# Default target
all: $(TARGET)
