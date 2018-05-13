#include<iostream>
#include "dataReader.h"
#include<float.h>
#include<math.h>
#include<numeric>
#include<algorithm>
#include<sys/time.h>

using namespace std;

#define FILENAME "clean2.data"
#define R_VALUE 3
//#define debug

__global__ void calEucledianDist( int *distMatrix, int columnSize, float *euclDistMatrix, int numOfRows)
{
	//int i = blockcounter;
	int i = blockDim.x*blockIdx.x+threadIdx.x;
	int j = blockDim.y*blockIdx.y+threadIdx.y;

	if(i>=numOfRows || j>=numOfRows)
		return;

	int index = i*numOfRows + j;
	int mirrorIndex = j*numOfRows + i;

	float euclideanDist = 0;
	for(int col=0; col<columnSize; col++)
		euclideanDist += pow(distMatrix[i*columnSize + col]-distMatrix[j*columnSize + col], 2); 
	
	euclideanDist = sqrt(euclideanDist);

	euclDistMatrix[index] = euclideanDist;
	euclDistMatrix[mirrorIndex] = euclideanDist;
}


__global__ void calHousdroffDist( int columnSize, int *bucketLocArr, int *bucketSizeArr, float *housdroffDistMatrix, float *euclDistMatrix, int euclDistColSize)
{
	int i = blockIdx.x;
	int j = threadIdx.x;

	#ifdef debug
	printf("Thread: %dx%d starting!\n", i, j);
	#endif

	if(i == j)
	{
		housdroffDistMatrix[i*columnSize + i] = FLT_MAX;
		return;
	}

	if(i > j)
		return;
	
	//calculating housdroff Distance between two bags
	//i represents bucket A while j represents bucket B
	//below for loops are similar to function call dist(A,B)
	//which returns housdroff distance between bucket A and B
	
	float d_A_B = FLT_MIN;
	float d_B_A = FLT_MIN;

	#ifdef debug
	printf("Thread: %dx%d going into bucket loop:\n", i, j);//debug
	#endif

	int bucketA_counter_initialize = bucketLocArr[i];
	int bucketA_counter_end = bucketLocArr[i]+bucketSizeArr[i];

	int bucketB_counter_initialize = bucketLocArr[j];
	int bucketB_counter_end = bucketLocArr[j]+bucketSizeArr[j];

	for(int bucketA_counter = bucketA_counter_initialize; bucketA_counter<bucketA_counter_end; bucketA_counter++)
	{
		float d_ai_B = FLT_MAX;

		for(int bucketB_counter = bucketB_counter_initialize; bucketB_counter<bucketB_counter_end; bucketB_counter++)
		{
			//calculate euclidean distance between rows represented by 
			//bucketA_counter and bucketB_counter
			float euclideanDist = euclDistMatrix[bucketA_counter*euclDistColSize + bucketB_counter];
			d_ai_B = min(d_ai_B, euclideanDist);
		}

		d_A_B = max(d_A_B, d_ai_B);
	}

	#ifdef debug
	printf("Thread: %dx%d going into d_bi_A loop:\n", i, j);//debug
	#endif

	for(int bucketB_counter = bucketB_counter_initialize; bucketB_counter<bucketB_counter_end; bucketB_counter++)
	{
		float d_bi_A = FLT_MAX;
		
		for(int bucketA_counter = bucketA_counter_initialize; bucketA_counter<bucketA_counter_end; bucketA_counter++)
		{
			//calculate euclidean distance between rows represented by 
			//bucketA_counter and bucketB_counter
			float euclideanDist = euclDistMatrix[bucketB_counter*euclDistColSize + bucketA_counter];
			d_bi_A = min(d_bi_A, euclideanDist);
		}

		d_B_A = max(d_B_A, d_bi_A);
	}

	float H_A_B = max(d_A_B, d_B_A);
	housdroffDistMatrix[i*columnSize + j] = H_A_B;
	housdroffDistMatrix[j*columnSize + i] = H_A_B;

	#ifdef debug
	printf("Thread: %d x %d done!\n", i, j);//debug
	#endif
}



//R is number of references
//C is number of citors
__global__ void calculateVoterList(int queryBag, int numOfBuckets, float *housdroffDistMatrix, int *classLabel, int *voterList, int R, int C)
{
	int voterListVal = 0;
	int closestNeighbourIndices[10];
	float maxVal = FLT_MIN;
	int maxClosestNeighbourIdx;
	int bucketCounter = threadIdx.x;
	int X;

	if(bucketCounter == queryBag)
	{
		X = R;
	}
	else
	{
		X = C;
	}

	for(int i=0; i<X; i++)
	{
		closestNeighbourIndices[i] = i;
		if(maxVal < housdroffDistMatrix[bucketCounter*numOfBuckets + i])
		{
			maxVal = housdroffDistMatrix[bucketCounter*numOfBuckets + i];
			maxClosestNeighbourIdx = i;
		}
	}

	for(int i=X; i<numOfBuckets; i++)
	{
		if(maxVal > housdroffDistMatrix[bucketCounter*numOfBuckets + i])
		{
			closestNeighbourIndices[maxClosestNeighbourIdx] = i;
			float tempMax = FLT_MIN;

			for(int j=0; j<X; j++)
			{
				if(tempMax < housdroffDistMatrix[bucketCounter*numOfBuckets + closestNeighbourIndices[j]])
				{
					tempMax = housdroffDistMatrix[bucketCounter*numOfBuckets + closestNeighbourIndices[j]];
					maxClosestNeighbourIdx = j;
				}
			}

			maxVal = tempMax;
		}
	}

	if(bucketCounter == queryBag)
	{
		for(int i=0; i<X; i++)
		{
			if(classLabel[closestNeighbourIndices[i]] == 1)
				voterListVal++;
			else
				voterListVal--;
		}
	}
	else
	{
		for(int i=0; i<X; i++)
		{
			if(queryBag == closestNeighbourIndices[i])
			{
				if(classLabel[closestNeighbourIndices[i]] == 1)
					voterListVal++;
				else
					voterListVal--;

				break;
			}
		}
	}

	voterList[bucketCounter] = voterListVal;
}

int main()
{
	testDataSet obj;
	obj.readData(FILENAME);
	
	int R = R_VALUE;
	int C = R + 2;

	float correctPrediction = 0;
	int numOfBuckets = obj.bucketSizeArr.size();

	#ifdef debug
	cout<<"total rows: "<<(obj.distMatrix.size())/obj.columnSize<<endl;
	cout<<"total columns: "<<obj.columnSize<<endl;
	#endif

	float *housdroffDistMatrix = (float *) malloc(sizeof(float) * numOfBuckets * numOfBuckets);

	#ifdef debug	
	cout<<"obj.distMatrix.size() "<<obj.distMatrix.size()<<endl;
	#endif

	int *d_distMatrix;
	if(!cudaMalloc(&d_distMatrix, sizeof(int) * obj.distMatrix.size()) == cudaSuccess)
		cout<<"error in allocating distMatrix\n";
	if(!cudaMemcpy(d_distMatrix, obj.distMatrix.data(), sizeof(int) * obj.distMatrix.size(), cudaMemcpyHostToDevice) == cudaSuccess)
		cout<<"error in copying distMatrix\n";

	int numOfRows = (obj.distMatrix.size())/obj.columnSize;
	cout<<"numRows"<<numOfRows<<endl;
	
	float *d_euclDistMatrix;
	if(!cudaMalloc(&d_euclDistMatrix, sizeof(float) * numOfRows * numOfRows) == cudaSuccess)
		cout<<"error in allocating d_euclDistMatrix\n";

	//for(int blockcounter = 0; blockcounter<numOfRows; blockcounter++)
	//	calEucledianDist<<<1,numOfRows>>>(d_distMatrix, obj.columnSize, d_euclDistMatrix, blockcounter);
	//calEucledianDist<<<numOfRows,numOfRows>>>(d_distMatrix, obj.columnSize, d_euclDistMatrix);

	dim3 threadDimension(32,32);
	dim3 blockDimension(numOfRows/32 + 1, numOfRows/32 + 1);
	calEucledianDist<<<blockDimension, threadDimension>>>(d_distMatrix, obj.columnSize, d_euclDistMatrix, numOfRows);

	cudaDeviceSynchronize();

	cudaFree(d_distMatrix);

#if 0
	//debug code start
	float *h_euclDistMatrix = (float *) malloc(sizeof(float) * numOfRows * numOfRows);
	cudaMemcpy(h_euclDistMatrix, d_euclDistMatrix, sizeof(float) * numOfRows * numOfRows, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	for(int i=0; i<numOfRows; i++)
	{
		for(int j=0; j<numOfRows; j++)
			printf("%f ", h_euclDistMatrix[i*numOfRows+j]);

		printf("\n");
	}

	free(h_euclDistMatrix);
	//debug code end
#endif

	int *d_bucketLocArr;
	if(!cudaMalloc(&d_bucketLocArr, sizeof(int) * obj.bucketLocArr.size())  == cudaSuccess)
		cout<<"error in allocating d_bucketLocArr\n";
	if(!cudaMemcpy(d_bucketLocArr, obj.bucketLocArr.data(), sizeof(int) * obj.bucketLocArr.size(), cudaMemcpyHostToDevice) == cudaSuccess)
		cout<<"error in copying d_bucketLocArr\n";
	
	int *d_bucketSizeArr;
	if(!cudaMalloc(&d_bucketSizeArr, sizeof(int) * obj.bucketSizeArr.size()) == cudaSuccess)
		cout<<"error in allocating d_bucketSizeArr\n";
	if(!cudaMemcpy(d_bucketSizeArr, obj.bucketSizeArr.data(), sizeof(int) * obj.bucketSizeArr.size(), cudaMemcpyHostToDevice) == cudaSuccess)
		cout<<"error in copying d_bucketSizeArr\n";

	float *d_housdroffDistMatrix;
	if(!cudaMalloc(&d_housdroffDistMatrix, sizeof(float) * numOfBuckets * numOfBuckets) == cudaSuccess)
		cout<<"error in allocating d_housdroffDistMatrix\n";

	cout<<"malloc and cudaMemcpy done\n";

	calHousdroffDist<<<numOfBuckets,numOfBuckets>>>(obj.columnSize, d_bucketLocArr, d_bucketSizeArr, d_housdroffDistMatrix, d_euclDistMatrix, numOfRows);
	cudaDeviceSynchronize();
	
	cout<<"housdroff distance calculation complete\n";

#if 0
	//debug code start
	cout<<"\n\nprinting housdroff matrix\n\n";
	float *h_housdroffDistMatrix = (float *) malloc(sizeof(float) * numOfBuckets * numOfBuckets);
	cudaMemcpy(h_housdroffDistMatrix, d_housdroffDistMatrix, sizeof(float) * numOfBuckets * numOfBuckets, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	for(int i=0; i<numOfBuckets; i++)
	{
		for(int j=0; j<numOfBuckets; j++)
			printf("%f ", h_housdroffDistMatrix[i*numOfBuckets+j]);

		printf("\n");
	}

	free(h_housdroffDistMatrix);
	//debug code end
#endif

	cudaFree(d_bucketLocArr);
	cudaFree(d_bucketSizeArr);
	cudaFree(d_euclDistMatrix);

	int *d_classLabel;
	if(!cudaMalloc(&d_classLabel, sizeof(int) * obj.classLabel.size()) == cudaSuccess)
		cout<<"error in allocating d_classLabel\n";
	if(!cudaMemcpy(d_classLabel, obj.classLabel.data(), sizeof(int) * obj.classLabel.size(), cudaMemcpyHostToDevice) == cudaSuccess)
		cout<<"error in copying d_classLabel\n";

	int *d_voterList;
	if(!cudaMalloc(&d_voterList, sizeof(int) * numOfBuckets) == cudaSuccess)
		cout<<"error in allocating d_voterList\n";

	int *h_voterList = (int*) malloc(sizeof(int) * numOfBuckets);

	for(int queryBag=0; queryBag<numOfBuckets; queryBag++)
	{
		cudaMemset(d_voterList, 0, sizeof(int) * numOfBuckets);
		int classValue = 0;

		calculateVoterList<<<1,numOfBuckets>>>(queryBag, numOfBuckets, d_housdroffDistMatrix, d_classLabel, d_voterList, R, C);
		cudaDeviceSynchronize();
		
		cudaMemcpy(h_voterList, d_voterList, sizeof(int) * numOfBuckets, cudaMemcpyDeviceToHost);
		
		for(int j=0; j<numOfBuckets; j++)
			classValue += h_voterList[j];
		
		if(classValue > 0)
			classValue = 1;
		else
			classValue = 0;

		if(classValue == obj.classLabel[queryBag])
			correctPrediction++;
	}

	cout<<"correctPrediction = "<<correctPrediction<<endl;
	cout<<"dataSize = "<<numOfBuckets<<endl;
	float accuracy = (correctPrediction * 100.0)/(float) numOfBuckets;
	cout<<"accurary is "<<accuracy<<endl;

	cudaFree(d_housdroffDistMatrix);
	cudaFree(d_classLabel);
	cudaFree(d_voterList);

	free(h_voterList);
	free(housdroffDistMatrix);

	return 0;
}