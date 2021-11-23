CFLAGS = $(shell pkg-config --cflags opencv4)
LIBS = $(shell pkg-config --libs opencv4)

hw8.out : hw8.cpp
	clang++ $(CFLAGS) $(LIBS) -o $@ $<

test:
	clang++ $(CFLAGS) $(LIBS) -o test test.cpp

clean:
	rm -f *.out