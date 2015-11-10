CC=g++
CFLAGS += -O3 -g 
CXXFLAGS += -std=c++11 -O3 -fopenmp -g
LDFLAGS += -static -fopenmp
LDLIBS +=  -lxgboost /usr/local/lib/dmlc_simple.o -lrabit -Wl,--whole-archive -lcaffe -Wl,--no-whole-archive -lproto -lprotobuf -lsnappy -lgflags -lglog -lleveldb -llmdb -lunwind -lhdf5_hl -lhdf5 -lopencv_features2d -lopencv_imgproc -lopencv_imgcodecs -lopencv_flann -lopencv_core -lopencv_hal -lIlmImf -lippicv -lboost_timer -lboost_chrono -lboost_program_options -lboost_log -lboost_log_setup -lboost_thread -lboost_filesystem -lboost_system -lopenblas -ljpeg -ltiff -lpng -ljasper -lwebp -lpthread -lz -lm -lrt -ldl

PROGS = caffex-extract	caffex-predict batch-resize

all:	$(PROGS)

caffex-extract:	caffex-extract.cpp caffex.cpp

caffex-predict:	caffex-predict.cpp caffex.cpp

caffex-compare:	caffex-compare.cpp caffex.cpp

batch-resize:	batch-resize.cpp
