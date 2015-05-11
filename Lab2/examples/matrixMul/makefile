CC	:= g++
NVCC	:= nvcc
LINKER	:= g++
INCLUDES	:= -I.
LIBS		:= 
CUDA_LIBS	:= $(LIBS) -L"$(CUDA_INSTALL_PATH)/lib64"
NVCCFLAGS	:= --ptxas-options=-v
#NVCCFLAGS	:= -arch compute_20 -code sm_20 --ptxas-options=-v
#NVCCFLAGS	:= -arch compute_11 -code sm_11 --ptxas-options=-v
LDFLAGS         := -lm -lX11 -lpthread
CUDA_LDFLAGS	:= $(LDFLAGS) -lrt -lcudart

debug		:= 0
ifneq ($(debug), 1)
	CFLAGS		:= -O3 -g0 -w
	NVCCFLAGS	+= -O3
else
	CFLAGS		:= -O0 -g3 -Wall
	NVCCFLAGS	+= -O0
endif

################################################################################
# CUDA code generation flags
#GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30
#GENCODE_SM32    := -gencode arch=compute_32,code=sm_32
#GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
GENCODE_SM50    := -gencode arch=compute_50,code=sm_50
GENCODE_SMXX    := -gencode arch=compute_50,code=compute_50
GENCODE_FLAGS   ?= $(GENCODE_SM10) $(GENCODE_SM20) $(GENCODE_SM30) \
                $(GENCODE_SM32) $(GENCODE_SM35) $(GENCODE_SM50) $(GENCODE_SMXX)

NVCCFLAGS+= $(GENCODE_FLAGS)

CUSRCS=matrixMul.cu
CPPSRCS=Timer.cpp
OBJS=$(CUSRCS:%.cu=%.o)
OBJS+=$(CPPSRCS:%.cpp=%.o)

EXEC=exec
RUN_CMD=./$(EXEC)

all: $(EXEC) 

$(EXEC): $(OBJS)
	$(LINKER) $(OBJS) -o $(EXEC) $(CUDA_LIBS) $(CUDA_LDFLAGS)

%.o: %.cu
	$(NVCC) $(INCLUDES) $(NVCCFLAGS)  -c $< -o $@

%.o: %.cpp
	$(CC) $(INCLUDES) $(CFLAGS) -c $< -o $@

run:	all
	$(RUN_CMD)
	
clean:
	rm -f $(EXEC) $(OBJS) *~ 

