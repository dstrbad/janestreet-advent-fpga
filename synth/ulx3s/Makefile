# Makefile for ULX3S synthesis
# Requires: yosys, nextpnr-ecp5, ecppack (from prjtrellis)

# Configuration
DEVICE = 85k
PACKAGE = CABGA381
SPEED = 6

# Source files
TOP = top
VERILOG = top.v ../../rtl/inference_engine.v
CONSTRAINTS = ulx3s_v20.lpf

# Output files
JSON = $(TOP).json
CONFIG = $(TOP).config
BITSTREAM = $(TOP).bit
SVF = $(TOP).svf

# Tools - use local OSS CAD Suite for ECP5 tools
OSS_CAD_SUITE = ../../oss-cad-suite
YOSYS = yosys
NEXTPNR = $(OSS_CAD_SUITE)/bin/nextpnr-ecp5
ECPPACK = $(OSS_CAD_SUITE)/bin/ecppack
FUJPROG = fujprog

.PHONY: all synth pnr bitstream program clean

all: bitstream

# Synthesis with Yosys
synth: $(JSON)

$(JSON): $(VERILOG)
	$(YOSYS) -p "read_verilog $(VERILOG); synth_ecp5 -top $(TOP) -json $@"

# Place and Route with nextpnr
pnr: $(CONFIG)

$(CONFIG): $(JSON) $(CONSTRAINTS)
	$(NEXTPNR) --$(DEVICE) --package $(PACKAGE) --speed $(SPEED) \
		--json $(JSON) --lpf $(CONSTRAINTS) \
		--textcfg $@ \
		--lpf-allow-unconstrained

# Generate bitstream
bitstream: $(BITSTREAM)

$(BITSTREAM): $(CONFIG)
	$(ECPPACK) --compress --input $< --bit $@

# Generate SVF for JTAG programming
svf: $(SVF)

$(SVF): $(CONFIG)
	$(ECPPACK) --compress --input $< --svf $@

# Program the FPGA via fujprog
program: $(BITSTREAM)
	$(FUJPROG) $<

# Program to flash (persistent)
flash: $(BITSTREAM)
	$(FUJPROG) -j flash $<

# Clean build artifacts
clean:
	rm -f $(JSON) $(CONFIG) $(BITSTREAM) $(SVF) *.log

# Show resource utilization (run after pnr)
utilization: $(CONFIG)
	@echo "=== Resource Utilization ==="
	@grep -E "^Info: " nextpnr.log 2>/dev/null | grep -E "(LUT|FF|BRAM|DSP)" || echo "Run 'make pnr' first"

# Quick synthesis check (no PnR)
check: $(JSON)
	@echo "Synthesis successful!"
	@$(YOSYS) -p "read_json $(JSON); stat"

# Help
help:
	@echo "ULX3S Synthesis Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build bitstream (default)"
	@echo "  synth     - Run Yosys synthesis only"
	@echo "  pnr       - Run place and route"
	@echo "  bitstream - Generate bitstream"
	@echo "  program   - Program FPGA (volatile)"
	@echo "  flash     - Program to flash (persistent)"
	@echo "  check     - Quick synthesis check"
	@echo "  clean     - Remove build artifacts"
	@echo ""
	@echo "Requirements:"
	@echo "  - yosys (synthesis)"
	@echo "  - nextpnr-ecp5 (place and route)"
	@echo "  - ecppack from prjtrellis (bitstream generation)"
	@echo "  - fujprog (programming)"
