SRC := $(wildcard *.cc)
OBJ := $(SRC:.cc=.o)
EXE ?= faces

default: $(EXE)

$(EXE): $(OBJ)
	$(LD) $(LDFLAGS) $(OBJ) -o $(EXE) $(LIBS)

main.o: main.cc Array.h DArray.h Faces.h gpu.h Mugs.h

Mugs.o: Mugs.cc Mugs.h Array.h gpu.h

Faces.o: Faces.cc Faces.h DArray.h gpu.h

clean:
	rm -f $(OBJ) $(EXE)
