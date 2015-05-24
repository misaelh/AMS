nvcc -o main_cuda.o -c main.cu -O3

g++ -c -o GPUdb.o GPUdb.cpp -I"/opt/cuda/cuda50/include"  -L"/opt/cuda/cuda50/lib64" -O3 -g0  -lm -lX11 -lpthread -lrt -lcudart
g++ -c -o stdafx.o stdafx.cpp -I"/opt/cuda/cuda50/include"  -L"/opt/cuda/cuda50/lib64" -O3 -g0  -lm -lX11 -lpthread -lrt -lcudart
g++ -c -o GPUdbConverter.o GPUdbConverter.cpp -I"/opt/cuda/cuda50/include"  -L"/opt/cuda/cuda50/lib64" -O3 -g0  -lm -lX11 -lpthread -lrt -lcudart
g++ -c -o FastaFile.o FastaFile.cpp -I"/opt/cuda/cuda50/include"  -L"/opt/cuda/cuda50/lib64" -O3 -g0  -lm -lX11 -lpthread -lrt -lcudart
g++ -c -o CPU.o CPU.cpp -I"/opt/cuda/cuda50/include"  -L"/opt/cuda/cuda50/lib64" -O3 -g0  -lm -lX11 -lpthread -lrt -lcudart
g++ -c -o CommandLine.o CommandLine.cpp -I"/opt/cuda/cuda50/include"  -L"/opt/cuda/cuda50/lib64" -O3 -g0  -lm -lX11 -lpthread -lrt -lcudart
g++ -c -o FastaMatrix.o  FastaMatrix.cpp -I"/opt/cuda/cuda50/include"  -L"/opt/cuda/cuda50/lib64" -O3 -g0  -lm -lX11 -lpthread -lrt -lcudart -fpermissive

g++ -o gasw main.cpp main_cuda.o GPUdb.o stdafx.o GPUdbConverter.o FastaFile.o CPU.o CommandLine.o FastaMatrix.o -I"/opt/cuda/cuda50/include"  -L"/opt/cuda/cuda50/lib64" -O3 -g0  -lm -lX11 -lpthread -lrt -lcudart


