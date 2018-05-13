all: dataReader.cpp citationKNN.cpp 
	g++ -std=c++11 -g -o citationKNN dataReader.cpp citationKNN.cpp
	nvcc -std=c++11 -g -o citationKNN_cu dataReader.cpp citationKNN_cuda.cu

clean: 
	rm citationKNN
	rm citationKNN_cu