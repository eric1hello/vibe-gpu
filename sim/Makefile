MODULE=gpu_top
RTL_DIR=../rtl
SIM_DIR=.

.PHONY: all clean run

all: obj_dir/V$(MODULE)

obj_dir/V$(MODULE): $(SIM_DIR)/main.cpp $(RTL_DIR)/*.sv
	verilator -cc --exe --build -j 4 -Wall \
		-I$(RTL_DIR) \
		--trace \
		$(RTL_DIR)/$(MODULE).sv \
		$(SIM_DIR)/main.cpp \
		-o V$(MODULE)

run: all
	./obj_dir/V$(MODULE)

clean:
	rm -rf obj_dir sim_trace.vcd

