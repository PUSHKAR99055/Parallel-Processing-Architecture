NC = nvcc

CC = g++

CFLAGS = 

SRC1g2m = cuda-1g-wsm2.cu

OBJ_1g2m = cuda-1g-wsm2.o

SRC_SSSP = sssp.cu

OUT_SSSP = sssp.out

SRC_VIZ = sssp-visualize.cpp

OBJ_VIZ = sssp-visualize.o


sssp:$(OBJ_1g2m) $(OBJ_VIZ) $(SRC_SSSP) 
	$(NC) $(CFLAGS) $^ -o $(OUT_SSSP)  `pkg-config --cflags --libs opencv4`

$(OBJ_1g2m):$(SRC1g2m)
	$(NC) $(CFLAGS) -c $^ -o $@

$(OBJ_VIZ):$(SRC_VIZ)
	$(CC) -c $^ -o $@ -I/usr/local/include/opencv4/

run:sssp
	./$(OUT_SSSP)

clean:
	rm $(OUT_SSSP) $(OBJ_1g2m) $(OBJ_VIZ)
