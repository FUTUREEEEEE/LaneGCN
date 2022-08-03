conda activate pt12_cu113_munet_env 

CUDA_V=11.3

export PATH=/mnt/cache/share/cuda-${CUDA_V}/bin:$PATH

export LD_LIBRARY_PATH=/mnt/cache/share/cuda-${CUDA_V}/lib64:$LD_LIBRARY_PATH

nvcc -V