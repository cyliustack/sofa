TARGET := sofa
OBJS := main.o
LIB_USER := 
LIB_USER_OBJS :=  
OPTIONS := -g 
CFLAGS := $(OPTIONS)
CXXFLAGS := $(OPTIONS)
CC := gcc
CXX := g++
all	: $(TARGET) $(LIB_USER)
$(TARGET): $(OBJS) $(LIB_USER)
	@$(CXX) -o $@ $(OBJS) $(LIB_USER)
	@echo "\033[0;33m Generated	$@\033[0m"
	@echo "\033[0;33m Generated	$@\033[0m"
$(LIB_USER):	$(LIB_USER_OBJS)
	@ar -rcs $@ $^
	@echo "\033[0;32m LINK	$@	by	$^\033[0m"
%.o	:	%.c
	@$(CC) -c -fPIC $(CFLAGS) $< -o $@
	@echo "\033[0;32m CC	$@\033[0m"
%.o	:	%.cpp
	@$(CXX) -c -fPIC $(CXXFLAGS) $< -o $@
	@echo "\033[0;32m CXX	$@\033[0m"
run: $(TARGET)
	./$(TARGET)
	python3 -i plot-all.py
debug: $(TARGET)
	gdb --args ./$(TARGET)
clean:
	@rm $(TARGET) $(LIB_USER) *.o
	@echo "\033[0;34m Removed all TARGET and objects\033[0m"
