build:	
	mkdir build; cd build; cmake ..; make

run:
	./build/main

clean:
	rm -fr build
