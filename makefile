# Compiler settings
CC = gcc
CXX = g++
CFLAGS = -Wall -O2
CXXFLAGS = -Wall -O2

# Source files
C_SOURCES = mnist_loader.c
CXX_SOURCES = main.cpp Mat.cpp

# Object files
C_OBJECTS = $(C_SOURCES:.c=.o)
CXX_OBJECTS = $(CXX_SOURCES:.cpp=.o)

# Executable name
EXECUTABLE = main.exe

# Default target
all: $(EXECUTABLE)

# Linking
$(EXECUTABLE): $(C_OBJECTS) $(CXX_OBJECTS)
	$(CXX) $(C_OBJECTS) $(CXX_OBJECTS) -o $@

# Compiling C files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Compiling C++ files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up
clean:
	del /Q *.o $(EXECUTABLE)

# Phony targets
.PHONY: all clean