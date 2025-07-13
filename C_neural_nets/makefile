# Makefile

CC = gcc
CFLAGS = -Wall -Wextra -Ih

SRC_DIR = src
OBJ_DIR = obj
BIN = otter

# List your source files (without extension or path) here
FILES = main ottertensors ottertensors_utilities ottertensors_operations ottertensors_random ottermath otternet otternet_optimizers otterLayers otternet_utilities OtterActivation

SRC = $(addprefix $(SRC_DIR)/,$(addsuffix .c,$(FILES)))
OBJ = $(addprefix $(OBJ_DIR)/,$(addsuffix .o,$(FILES)))

all: $(BIN)

$(BIN): $(OBJ)
	$(CC) $(OBJ) -o $(BIN)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(OBJ_DIR) $(BIN)

.PHONY: all clean
