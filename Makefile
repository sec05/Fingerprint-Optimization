# Define directories
BASE_DIR := .
OBJ_DIR := ./obj
FINGERPRINT_DIR := ./Fingerprints
STATE_DIR := ./State

# Collect all .cpp files in the base directory, NLA directory, Fingerprints directory, and State directory
SRC_FILES := $(wildcard $(BASE_DIR)/*.cpp) \
             $(wildcard $(FINGERPRINT_DIR)/*.cpp) \
             $(wildcard $(STATE_DIR)/*.cpp)

# Define the target executable
TARGET := fingerprint_optimizer

# Define the compiler and compiler flags
CXX := g++
CXXFLAGS := -std=c++17 -O2 -shared-libgcc -MMD -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
LDFLAGS := -Xpreprocessor -fopenmp -L/opt/homebrew/opt/libomp/lib -L/opt/homebrew/opt/openblas/lib -L/opt/homebrew/opt/lapack/lib -Wl,-rpath,/opt/homebrew/opt/armadillo/lib
LDLIBS :=  -lomp -larmadillo -lopenblas -llapack

# Define the object files (with paths relative to OBJ_DIR)
OBJ_FILES := $(patsubst $(BASE_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(wildcard $(BASE_DIR)/*.cpp)) \
             $(patsubst $(FINGERPRINT_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(wildcard $(FINGERPRINT_DIR)/*.cpp)) \
             $(patsubst $(STATE_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(wildcard $(STATE_DIR)/*.cpp))

# Define the dependency files
DEP_FILES := $(OBJ_FILES:.o=.d)

# Rule to create the obj directory if it doesn't exist
$(shell mkdir -p $(OBJ_DIR))

# Rule to build the target executable
$(TARGET): $(OBJ_FILES)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# Rule to build object files
$(OBJ_DIR)/%.o: $(BASE_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -MMD -c $< -o $@

$(OBJ_DIR)/%.o: $(FINGERPRINT_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -MMD -c $< -o $@

$(OBJ_DIR)/%.o: $(STATE_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -MMD -c $< -o $@

# Include dependency files
-include $(DEP_FILES)

# Clean target to remove generated files
clean:
	rm -rf $(OBJ_DIR) $(TARGET)

tridiag:
	make
	./fingerprint_optimizer
	python Matrix\ Output/validator.py

# Phony targets
.PHONY: all clean

# Default target
all: $(TARGET)