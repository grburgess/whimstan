# Builds CLOUDY c23.01 from source and provides grid-pipeline entry points.
# Everything heavyweight lives under this directory and is gitignored.

CLOUDY_VERSION := c23.01
CLOUDY_URL     := https://data.nublado.org/cloudy_releases/c23/$(CLOUDY_VERSION).tar.gz
CLOUDY_DIR     := $(CURDIR)/$(CLOUDY_VERSION)
CLOUDY_EXE     := $(CLOUDY_DIR)/source/cloudy.exe
export CLOUDY_DATA_PATH := $(CLOUDY_DIR)/data

NPROC := $(shell sysctl -n hw.ncpu 2>/dev/null || nproc)

.PHONY: cloudy smoke grid assemble clean-runs

cloudy: $(CLOUDY_EXE)

$(CLOUDY_EXE):
	curl -fL -o $(CLOUDY_VERSION).tar.gz $(CLOUDY_URL)
	tar xzf $(CLOUDY_VERSION).tar.gz
	$(MAKE) -C $(CLOUDY_DIR)/source -j$(NPROC)

smoke: $(CLOUDY_EXE)
	printf 'test\n' | $(CLOUDY_EXE) && echo "CLOUDY SMOKE OK"

grid: $(CLOUDY_EXE)
	cd ../.. && uv run python tools/build_whim_grid/run_grid.py \
	    --cloudy-exe $(CLOUDY_EXE) --out tools/build_whim_grid/runs

assemble:
	cd ../.. && uv run python tools/build_whim_grid/assemble.py \
	    --runs tools/build_whim_grid/runs --out tools/build_whim_grid/dist

clean-runs:
	rm -rf runs dist
