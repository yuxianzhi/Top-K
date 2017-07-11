CC=g++
CCFLAGS= -O3 -I . -I /usr/local/cuda-7.0/include
LDFLAGS= -O3 -L /usr/local/cuda-7.0/lib64/  -lcudart -lcuda
NVCC=nvcc
NVCCFLAGS= -O3 -I .

all: main
main: main.o benchmark.o top_k_cpu.o top_k_gpu.o util.o top_k_thrust.o
	@$(CC) -o main main.o benchmark.o top_k_cpu.o top_k_gpu.o util.o top_k_thrust.o $(LDFLAGS)
	@rm main.o benchmark.o top_k_cpu.o top_k_gpu.o util.o top_k_thrust.o
main.o: main.cpp
	@$(CC) $(CCFLAGS) -o main.o -c main.cpp
benchmark.o: benchmark.cpp
	@$(CC) $(CCFLAGS) -o benchmark.o -c benchmark.cpp
top_k_cpu.o: top_k_cpu.cpp
	@$(CC) $(CCFLAGS) -o top_k_cpu.o -c top_k_cpu.cpp
top_k_gpu.o: top_k_gpu.cu
	@$(NVCC) $(NVCCFLAGS) -o top_k_gpu.o -c top_k_gpu.cu
util.o: util.cpp
	@$(CC) $(CCFLAGS) -o util.o -c util.cpp
top_k_thrust.o: top_k_thrust.cu
	@$(NVCC) $(NVCCFLAGS) -o top_k_thrust.o -c top_k_thrust.cu

clean:
	rm main
