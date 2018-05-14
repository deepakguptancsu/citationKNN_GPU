#Accelerator-Based High Performance Machine Learning
Parallelizing citationKNN (machine learning algorithm) on GPU

CitationKNN - CitationKNN is one of the most popular algorithms for solving multiple-instance learning (MIL) problems. It is a lazy learning algorithm which tries to classify a bag of instances by using labelled bags of instances.

Its uses are diverse, like:
- Used for solving musky molecule prediction task 
- Used for image classification 
- Examining medical images to find tumors
- Web mining
- Spam detection
- Remote sensing
- Stock selection

CitationKNN's useability and effectiveness makes it popular. However, it involves very high data intensive calculations, which makes it a suitable candidate for parallelization.

Code Architecture for serial code (citationKNN.cpp):
dataReader.cpp - provide interfaces to read the input dataset in testDataSet datastructure  
calHousdroffDist() - then calculates the housdroff distances between individual bags
calClassLabel() - then calculates the citors and references of query bag, which further helps in calculating class label

This project parallelizes CitationKNN on GPU using CUDA. Parallelized sections include (in citationKNN_cuda.cu):
- calculating eucledian distances between individual instances
- calculating Hausdroff distance between bags containing instances
- calculating citors and references of query bag

Results:
(The data set used for testing is Musk2 dataset)
- The accuracy of parallelized algorithm is same as of serialized algorithm, given in original paper (Jun Wang and Jean-Daniel Zucker, "Solving multiple-instance problem: A lazy learning approach" (2000))
- A speedup of 45.5x was achieved on GTX1080 GPU
