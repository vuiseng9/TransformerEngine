clean:
	pip uninstall -y transformer_engine
	rm -rf build/ dist/ *.egg-info log.build
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.so" -delete

submodule-update:
	git submodule update --init --recursive

debug-build-install: submodule-update
	rm -f log.build
	NVTE_CUDA_ARCHS=100 NVTE_FRAMEWORK=pytorch NVTE_CMAKE_EXTRA_ARGS="-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS=-O0 -DCMAKE_CUDA_FLAGS=-O0" NVTE_TORCH_CUDA_FLAGS="-O0 -g" NVTE_BUILD_DEBUG=1 MAX_JOBS=8 pip install -e . -v --no-build-isolation 2>&1 | tee log.build

build-install: submodule-update
	rm -f log.build
	NVTE_CUDA_ARCHS=100 NVTE_FRAMEWORK=pytorch NVTE_CMAKE_EXTRA_ARGS="-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc" MAX_JOBS=8 pip install -e . -v --no-build-isolation 2>&1 | tee log.build
