clean:
	pip uninstall -y transformer_engine
	rm -rf build/ dist/ *.egg-info log.build
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.so" -delete

submodule-update:
	git submodule update --init --recursive

debug-build-install:
	rm -f log.build
	NVTE_CUDA_ARCHS=100 NVTE_FRAMEWORK=pytorch NVTE_CMAKE_EXTRA_ARGS="-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS=-O0 -DCMAKE_CUDA_FLAGS=-O0" NVTE_TORCH_CUDA_FLAGS="-O0 -g" NVTE_BUILD_DEBUG=1 MAX_JOBS=8 pip install -e . -v --no-build-isolation 2>&1 | tee log.build

build-install:
	rm -f log.build
	NVTE_CUDA_ARCHS=100 NVTE_FRAMEWORK=pytorch NVTE_CMAKE_EXTRA_ARGS="-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc" MAX_JOBS=8 pip install -e . -v --no-build-isolation 2>&1 | tee log.build

build-release:
	rm -f log.build
# 	git checkout v2.5+nvfp4+wip
# 	git reset c2056e01b7c1d458bb3a86e895c11eb401c181a0
# 	git checkout HEAD --  build_tools/pytorch.py
# 	git checkout HEAD -- transformer_engine/common/CMakeLists.txt
	NVTE_CUDA_ARCHS=100 NVTE_FRAMEWORK=pytorch NVTE_CMAKE_EXTRA_ARGS="-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc" MAX_JOBS=8 python setup.py bdist_wheel 2>&1 | tee log.build
# 	git clean -f
# 	git reset --hard
# 	git pull

verify-te:
	pip show transformer_engine
	PYTHONPATH=/root/work/dev/f4/qt python /root/work/dev/f4/qt/scripts/05_mxfp8_block_scaling.py
