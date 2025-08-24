CC = gcc
CFLAGS = -Wall -Wextra -g -fsanitize=address,undefined -fno-omit-frame-pointer -Ih -lm

SRC_DIR = src
OBJ_DIR = obj
BIN = otter

FILES = main ottertensors ottertensors_utilities ottertensors_operations ottertensors_random \
        ottermath otternet otternet_optimizers OtterLayers otternet_utilities OtterActivation OtterDisplay

SRC = $(addprefix $(SRC_DIR)/,$(addsuffix .c,$(FILES)))
OBJ = $(addprefix $(OBJ_DIR)/,$(addsuffix .o,$(FILES)))

all: $(BIN)

$(BIN): $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o $(BIN)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(OBJ_DIR) $(BIN)

.PHONY: all clean