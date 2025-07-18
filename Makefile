SRC = main.cpp
OBJ = $(SRC:.cpp=.o)
DEPS = math.hpp

CC = g++

COMPILER_FLAGs = -std=c++17 -Wall

LINKER_FLAGS = -lSDL2

TARGET = raytracer

all : $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(OBJ) $(LINKER_FLAGS) -o $@

%.o: %.cpp $(DEPS)
	$(CC) -c $< $(CFLAGS)

clean:
	rm -f *.o $(TARGET)
