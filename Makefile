C_NONE=$(shell echo -e "\033[0m")
C_RED=$(shell echo -e "\033[31m")
C_GREEN=$(shell echo -e "\033[32m")
C_ORANGE=$(shell echo -e "\033[33m")
C_BLUE=$(shell echo -e "\033[34m")
C_PURPLE=$(shell echo -e "\033[35m")
C_CYAN=$(shell echo -e "\033[36m")
C_LIGHT_GRAY=$(shell echo -e "\033[37m")

TARGET := fsa
OBJS := main.o
LIB_USER := 
LIB_USER_OBJS :=  
OPTIONS := -g -std=c++14 
CFLAGS := $(OPTIONS)
CXXFLAGS := $(OPTIONS)
LINKFLAGS := -lconfig++ -lpcap
CC := gcc
CXX := g++
all	: $(TARGET) $(LIB_USER)
$(TARGET): $(OBJS) $(LIB_USER)
	@$(CXX) -o $@ $(OBJS) $(LIB_USER) $(LINKFLAGS)
	@echo -e "$(C_ORANGE)Generated	$@$(C_NONE)"
	@echo -e "$(C_ORANGE)Generated	$@$(C_NONE)"
$(LIB_USER):	$(LIB_USER_OBJS)
	@ar -rcs $@ $^
	@echo -e "$(C_GREEN)LINK	$@	by	$^$(C_NONE)"
%.o	:	%.c
	@$(CC) -c -fPIC $(CFLAGS) $< -o $@
	@echo -e "$(C_GREEN)CC	$@$(C_NONE)"
%.o	:	%.cpp
	@$(CXX) -c -fPIC $(CXXFLAGS) $< -o $@
	@echo -e "$(C_GREEN)CXX	$@$(C_NONE)"
run: $(TARGET)
	rm -f perf.data
	perf record -a sleep 5
	perf script > perf.script
	./$(TARGET) default.cfg 
	cp report.js potatoboard
	python -mwebbrowser potatoboard/index.html
debug: $(TARGET)
	gdb --args ./$(TARGET)
install: uninstall
	mkdir -p /opt/sofa/bin
	mkdir -p /opt/sofa/potatoboard
	mkdir -p /opt/sofa/plugin
	cp -i fsa /opt/sofa/bin
	cp -i sofa /opt/sofa/bin
	cp -i potato /opt/sofa/bin
	cp potatoboard/app.js /opt/sofa/potatoboard
	cp potatoboard/index.html /opt/sofa/potatoboard
	ln -is /opt/sofa/bin/fsa /usr/local/bin/fsa
	ln -is /opt/sofa/bin/sofa /usr/local/bin/sofa
	ln -is /opt/sofa/bin/potato /usr/local/bin/potato
uninstall:
	rm -rf /opt/sofa
	rm -f /usr/local/bin/fsa
	rm -f /usr/local/bin/sofa
	rm -f /usr/local/bin/potato
clean:
	@rm $(TARGET) $(LIB_USER) *.o
	@echo -e "$(C_BLUE)Removed all TARGET and objects$(C_NONE)"
