# Compiler settings
CC = gcc
CXX = g++
CFLAGS = -Wall -O3 -g -I./include
CXXFLAGS = -Wall -O3 -g -std=c++14 -I./include -I. -D_USE_MATH_DEFINES

# Directories
SRC_DIR = src
TARGET_DIR = target
LIB_DIR = lib
BIN_DIR = bin

# Source files
C_SOURCES = $(wildcard $(SRC_DIR)/*.c)
CXX_SOURCES = $(wildcard $(SRC_DIR)/*.cpp)

# Object files
C_OBJECTS = $(patsubst $(SRC_DIR)/%.c,$(TARGET_DIR)/%.o,$(C_SOURCES))
CXX_OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(TARGET_DIR)/%.o,$(CXX_SOURCES))

# Executable name
EXECUTABLE = $(TARGET_DIR)/main.exe

# BLAS library
BLAS_LIB = $(LIB_DIR)/libopenblas.a
BLAS_DLL = $(BIN_DIR)/libopenblas.dll

# Default target
all: $(TARGET_DIR) $(EXECUTABLE)

# Create target directory
$(TARGET_DIR):
	mkdir $(TARGET_DIR)

# Linking
$(EXECUTABLE): $(C_OBJECTS) $(CXX_OBJECTS)
	$(CXX) $(C_OBJECTS) $(CXX_OBJECTS) -o $@ -L$(LIB_DIR) -l:libopenblas.a -lm

# Compiling C files
$(TARGET_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Compiling C++ files
$(TARGET_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Copy DLL to target directory
copy_dll: $(EXECUTABLE)
	copy $(BLAS_DLL) $(TARGET_DIR)

# Clean up
clean:
	if exist $(TARGET_DIR) rmdir /s /q $(TARGET_DIR)

# Build and run
run: $(EXECUTABLE)
	$(EXECUTABLE)

# Phony targets
.PHONY: all clean copy_dll