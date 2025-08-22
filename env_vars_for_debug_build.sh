# export NVTE_CUDA_ARCHS="90;100;120" # Hopper & Blackwell
export NVTE_CUDA_ARCHS="100" # Hopper & Blackwell
export NVTE_FRAMEWORK=pytorch 
export NVTE_CMAKE_EXTRA_ARGS="-DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS=-O0 -DCMAKE_CUDA_FLAGS=-O0"
export NVTE_TORCH_CUDA_FLAGS="-O0 -g"
export NVTE_BUILD_DEBUG=1

env | grep NVTE
env | grep NVTE | wc -l
echo ""
echo "make sure we have get the submodules"
echo "git submodule update --init --recursive"
echo ""
echo "Check the number of NVTE environment variables above, should have count of 5"
echo "MAX_JOBS=4 python setup.py bdist_wheel"
echo "or"
echo "MAX_JOBS=4 pip install -e . -v --no-build-isolation 2>&1 | tee log.build &"
echo ""
