# Build the Lumen CUPTI injection library.
#
#   make                 -- build liblumen_cupti_inject.so
#   make print-struct    -- show which CUpti_ActivityKernel* version was probed
#   make clean
#
# Requires the CUDA toolkit's CUPTI headers and library, present in
# nvidia/cuda:*-devel images under $(CUDA_HOME)/extras/CUPTI.

CUDA_HOME ?= /usr/local/cuda
CUPTI_DIR ?= $(CUDA_HOME)/extras/CUPTI

CC      ?= cc
TARGET  := liblumen_cupti_inject.so
SRC     := lumen_cupti_inject.c

INCLUDES := -I$(CUPTI_DIR)/include -I$(CUDA_HOME)/include
LIBDIRS  := -L$(CUPTI_DIR)/lib64 -L$(CUDA_HOME)/lib64
CFLAGS   += -O2 -g -fPIC -Wall -Wextra -std=c11 $(INCLUDES)
LDFLAGS  += -shared $(LIBDIRS) -lcupti -Wl,-rpath,$(CUPTI_DIR)/lib64

# CUPTI's kernel activity record is versioned and gains fields with each CUDA
# release. Probe downward from the newest known version and take the first that
# compiles AND exposes the four latency-timestamp fields, so this builds
# unchanged across toolkits. `probe-struct.sh` does the work -- keeping it in a
# script rather than a $(shell) one-liner avoids quoting the probe's newlines
# through make.
#
# Override explicitly if the probe picks wrong:
#   make KERNEL_STRUCT=CUpti_ActivityKernel8
HERE := $(dir $(lastword $(MAKEFILE_LIST)))
KERNEL_STRUCT ?= $(shell CC='$(CC)' INCLUDES='$(INCLUDES)' sh $(HERE)probe-struct.sh)

ifeq ($(strip $(KERNEL_STRUCT)),)
$(warning Could not probe a CUpti_ActivityKernel* version -- defaulting to CUpti_ActivityKernel9.)
$(warning If the build fails, pass it explicitly: make KERNEL_STRUCT=CUpti_ActivityKernel8)
override KERNEL_STRUCT := CUpti_ActivityKernel9
endif

CFLAGS += -DLUMEN_CUPTI_KERNEL_STRUCT=$(KERNEL_STRUCT)

.PHONY: all clean print-struct
all: $(TARGET)

$(TARGET): $(SRC)
	@echo "==> kernel record struct: $(KERNEL_STRUCT)"
	$(CC) $(CFLAGS) $(SRC) -o $@ $(LDFLAGS)
	@echo "==> built $@"
	@echo "    run with: CUDA_INJECTION64_PATH=$(CURDIR)/$@ <lumen command>"

print-struct:
	@echo "$(KERNEL_STRUCT)"

clean:
	rm -f $(TARGET) /tmp/.lumen_cupti_probe.c
