INCLUDEFLAGS :=-I./src -I/usr/local/include -I/usr/include/hdf5/serial
COMMON_FLAGS := -DCPU_ONLY --std=gnu++11
CXX :=g++
LD :=g++
LD_FLAGS := -L/usr/lib -L/usr/local/lib -L/usr/lib -L/usr/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu/hdf5/serial -L.build_debug/lib -lglog -lgflags -lprotobuf -lboost_system -lboost_filesystem -lm -lhdf5_hl -lhdf5 -lleveldb -lsnappy -llmdb -lopencv_core -lopencv_highgui -lopencv_imgproc -lboost_thread -lstdc++ -lcblas -latlas
SOURCE := src/classification.cpp
SOURCE_DIR := ./
BUILD_DIR := build
TARGET := classify.bin
CXX_SRCS := $(shell find src -name "*.cpp" ! -name "test_fix.cpp")
OBJS := $(addprefix $(BUILD_DIR)/, ${CXX_SRCS:.cpp=.o})

CAFFE_ROOT := /home/lewis/caffe-gpu
MODEL := $(CAFFE_ROOT)/models/bvlc_reference_caffenet/deploy.prototxt
WEIGHTS := $(CAFFE_ROOT)/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel
MEAN := $(CAFFE_ROOT)/data/ilsvrc12/imagenet_mean.binaryproto
SYNSET := $(CAFFE_ROOT)/data/ilsvrc12/synset_words.txt
IMG := $(CAFFE_ROOT)/examples/images/cat.jpg

classification.o: $(SOURCE)
	@ echo $(OBJS)
	$(CXX) $< -c -g -o $@ $(INCLUDEFLAGS) $(COMMON_FLAGS)

fix.o:src/test_fix.cpp
	$(CXX) $< -c -o $@ $(INCLUDEFLAGS) --std=gnu++11

fix.bin:fix.o
	$(LD) $< -o $@
all: $(TARGET)

classify :
	@ echo $(MODEL) $(WEIGHTS) $(MEAN) $(SYNSET) $(IMG)
	@ ./$(TARGET) $(MODEL) $(WEIGHTS) $(MEAN) $(SYNSET) $(IMG)

$(TARGET): $(OBJS)
	@ echo LD $@
	@ $(LD) -g -o $@ $(OBJS) $(LD_FLAGS)

$(BUILD_DIR)/%.o: %.cpp
	@ mkdir -p build
	@ mkdir -p build/src
	@ mkdir -p build/src/caffe
	@ mkdir -p build/src/caffe/layers
	@ mkdir -p build/src/caffe/proto
	@ mkdir -p build/src/caffe/util
	@ mkdir -p build/src/caffe/solvers
	@ echo CXX $<
	@ $(CXX) $< -c -g -o $@ $(INCLUDEFLAGS) $(COMMON_FLAGS)

clean:
	@ rm -rf build
	@ $(shell find $(SOURCE_DIR) -name "*.o" |xargs rm -f)
	@ $(shell find $(SOURCE_DIR) -name "*.bin" |xargs rm -f)
