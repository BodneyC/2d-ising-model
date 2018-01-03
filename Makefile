CC=mpic++
LIBS=-g -fopenmp -O3
SRC=$(wildcard *.C)
BIN=$(patsubst %.C, %, $(SRC))

all: $(BIN)

$(BIN): $(SRC)
	$(CC) $(LIBS) -o $@ $<

.PHONY: clean 

clean:
	rm $(BIN)

