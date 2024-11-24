
MPICC=mpiicx
CC=icx
CFLAGS= -O2 -lm -qopt-report=1 -qopt-report -qopenmp
SRc = heat.c colormap.h stb_image_write.h
OBJ= heat.o

all:
	${MPICC} -o heat heat.c ${CFLAGS}

run:
	mpirun -np 5 ./heat