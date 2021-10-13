CFLAGS = $(shell pkg-config --cflags opencv4)
LIBS = $(shell pkg-config --libs opencv4)

hw3.out : hw3.cpp
	clang++ $(CFLAGS) $(LIBS) -o $@ $<

clean:
	rm -f *.out