CFLAGS = $(shell pkg-config --cflags opencv4)
LIBS = $(shell pkg-config --libs opencv4)

hw7.out : hw7.cpp
	clang++ $(CFLAGS) $(LIBS) -o $@ $<

clean:
	rm -f *.out