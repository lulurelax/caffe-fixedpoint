INCLUDE_PATH=/home/lewis/caffe-gpu/include/
SRC_PATH=/home/lewis/caffe-gpu/src/
DEST_PATH=/home/lewis/project/caffe-test/src/
if [ $# == 0 ]; then
  echo "empty input..."
else
  cp ${SRC_PATH}$1.cpp ${DEST_PATH}$1.cpp
  cp ${INCLUDE_PATH}$1.hpp ${DEST_PATH}$1.hpp
fi
