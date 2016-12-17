INCLUDEFLAGS :=-I./src -I/usr/local/include -I/usr/include/hdf5/serial
COMMON_FLAGS := -DCPU_ONLY --std=gnu++11
CXX :=g++
LD :=g++
SOURCE := src/classification.cpp

classification.o: $(SOURCE)
	$(CXX) $< -c -o $@ $(INCLUDEFLAGS) $(COMMON_FLAGS)

fix.o:src/test_fix.cpp
	$(CXX) $< -c -o $@ $(INCLUDEFLAGS) --std=gnu++11

fix:fix.o
	$(LD) $< -o $@
