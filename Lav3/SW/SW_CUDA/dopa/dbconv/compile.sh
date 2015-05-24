g++ -c -o GPUdb.o ../gpu/GPUdb.cpp -I"/opt/cuda/cuda50/include"  -L"/opt/cuda/cuda50/lib64" -O3 -g0  -lm -lX11 -lpthread -lrt -lcudart
g++ -c -o stdafx.o ../gpu/stdafx.cpp -I"/opt/cuda/cuda50/include"  -L"/opt/cuda/cuda50/lib64" -O3 -g0  -lm -lX11 -lpthread -lrt -lcudart
g++ -c -o GPUdbConverter.o ../gpu/GPUdbConverter.cpp -I"/opt/cuda/cuda50/include"  -L"/opt/cuda/cuda50/lib64" -O3 -g0  -lm -lX11 -lpthread -lrt -lcudart
g++ -c -o FastaFile.o ../gpu/FastaFile.cpp -I"/opt/cuda/cuda50/include"  -L"/opt/cuda/cuda50/lib64" -O3 -g0  -lm -lX11 -lpthread -lrt -lcudart
g++ -c -o CPU.o ../gpu/CPU.cpp -I"/opt/cuda/cuda50/include"  -L"/opt/cuda/cuda50/lib64" -O3 -g0  -lm -lX11 -lpthread -lrt -lcudart
g++ -c -o FastaMatrix.o  ../gpu/FastaMatrix.cpp -I"/opt/cuda/cuda50/include"  -L"/opt/cuda/cuda50/lib64" -O3 -g0  -lm -lX11 -lpthread -lrt -lcudart -fpermissive

g++ -o DBconv main.cpp GPUdb.o stdafx.o GPUdbConverter.o FastaFile.o CPU.o  FastaMatrix.o -I"/opt/cuda/cuda50/include"  -L"/opt/cuda/cuda50/lib64" -O3 -g0  -lm -lX11 -lpthread -lrt -lcudart


