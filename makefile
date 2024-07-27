# Compiler settings
CC = gcc
CXX = g++
CFLAGS = -Wall -O2
CXXFLAGS = -Wall -O2

# Directories
SRC_DIR = src
TARGET_DIR = target

# Source files
C_SOURCES = $(wildcard $(SRC_DIR)/*.c)
CXX_SOURCES = $(wildcard $(SRC_DIR)/*.cpp)

# Object files
C_OBJECTS = $(patsubst $(SRC_DIR)/%.c,$(TARGET_DIR)/%.o,$(C_SOURCES))
CXX_OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(TARGET_DIR)/%.o,$(CXX_SOURCES))

# Executable name
EXECUTABLE = $(TARGET_DIR)/main.exe

# Default target
all: $(TARGET_DIR) $(EXECUTABLE)

# Create target directory
$(TARGET_DIR):
	mkdir $(TARGET_DIR)

# Linking
$(EXECUTABLE): $(C_OBJECTS) $(CXX_OBJECTS)
	$(CXX) $(C_OBJECTS) $(CXX_OBJECTS) -o $@

# Compiling C files
$(TARGET_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Compiling C++ files
$(TARGET_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up
clean:
	if exist $(TARGET_DIR) rmdir /s /q $(TARGET_DIR)

# Build and run
run: $(EXECUTABLE)
	$(EXECUTABLE)

# Phony targets
.PHONY: all clean