#include<iostream>
#include "dataReader.h"
#include<float.h>
#include<math.h>
#include<numeric>
#include<algorithm>

using namespace std;

#define FILENAME "clean2.data"
#define R_VALUE 3
//#define debug

int calHousdroffDist( vector<int> &distMatrix, int columnSize, vector<int> &bucketLocArr, vector<int> &bucketSizeArr, float **housdroffDistMatrix)
{
	for(int i=0; i<bucketLocArr.size(); i++)
	{
		housdroffDistMatrix[i][i] = FLT_MAX;
	}

	for(int i=0; i<bucketLocArr.size()-1; i++)
	{
		for(int j=i+1; j<bucketLocArr.size(); j++)
		{
			//calculating housdroff Distance between two bags
			//i represents bucket A while j represents bucket B
			//below for loops are similar to function call dist(A,B)
			//which returns housdroff distance between bucket A and B
			
			float d_A_B = FLT_MIN;
			float d_B_A = FLT_MIN;

			for(int bucketA_counter = bucketLocArr[i]; bucketA_counter<bucketLocArr[i]+bucketSizeArr[i]; bucketA_counter++)
			{
				float d_ai_B = FLT_MAX;

				for(int bucketB_counter = bucketLocArr[j]; bucketB_counter<bucketLocArr[j]+bucketSizeArr[j]; bucketB_counter++)
				{
					//calculate euclidean distance between rows represented by 
					//bucketA_counter and bucketB_counter
					float euclideanDist = 0;
					for(int col=0; col<columnSize; col++)
						euclideanDist += pow(distMatrix[bucketA_counter*columnSize + col]-distMatrix[bucketB_counter*columnSize + col], 2); 
					
					euclideanDist = sqrt(euclideanDist);

					//cout<<euclideanDist<<" ";

					d_ai_B = min(d_ai_B, euclideanDist);
				}

				//cout<<endl;

				//d_A_B = max(d_A_B, d_ai_B);
				//cout<<"d_A_B "<<i<<"_"<<j<<" "<<d_A_B<<endl;
			}

			for(int bucketB_counter = bucketLocArr[j]; bucketB_counter<bucketLocArr[j]+bucketSizeArr[j]; bucketB_counter++)
			{
				float d_bi_A = FLT_MAX;
				
				for(int bucketA_counter = bucketLocArr[i]; bucketA_counter<bucketLocArr[i]+bucketSizeArr[i]; bucketA_counter++)
				{
					//calculate euclidean distance between rows represented by 
					//bucketA_counter and bucketB_counter
					float euclideanDist = 0;
					for(int col=0; col<columnSize; col++)
						euclideanDist += pow(distMatrix[bucketA_counter*columnSize + col]-distMatrix[bucketB_counter*columnSize + col], 2); 
					
					euclideanDist = sqrt(euclideanDist);

					d_bi_A = min(d_bi_A, euclideanDist);
				}

				d_B_A = max(d_B_A, d_bi_A);
				//cout<<"d_B_A "<<j<<"_"<<i<<" "<<d_B_A<<endl;
			}

			float H_A_B = max(d_A_B, d_B_A);
			housdroffDistMatrix[i][j] = H_A_B;
			housdroffDistMatrix[j][i] = H_A_B;
			//cout<<"H_"<<i<<"_"<<j<<" "<<H_A_B<<endl;
		}
	}
}

//R is number of references
//C is number of citors
vector<int> calculateVoterList(int queryBag, int numOfBuckets, float **housdroffDistMatrix, int R, int C)
{
	vector<int> voterList;
	for(int bucketCounter=0; bucketCounter<numOfBuckets; bucketCounter++)
	{
		vector<int> indices(numOfBuckets);
		iota(indices.begin(), indices.end(), 0); // fill with 0,1,2,...

		if(bucketCounter == queryBag)
		{
			partial_sort(indices.begin(), indices.begin()+R, indices.end(),
					[housdroffDistMatrix, bucketCounter](int x,int y) {return housdroffDistMatrix[bucketCounter][x]<housdroffDistMatrix[bucketCounter][y];});
						
			for(int i=0; i<R; i++)
				voterList.push_back(indices[i]);
		}
		else
		{
			partial_sort(indices.begin(), indices.begin()+C, indices.end(),
					[housdroffDistMatrix, bucketCounter](int x,int y) {return housdroffDistMatrix[bucketCounter][x]<housdroffDistMatrix[bucketCounter][y];});
			
			for(int i=0; i<C; i++)
			{
				if(queryBag == indices[i])
				{
					voterList.push_back(bucketCounter);
					break;
				}
			}
		}
	}

	return voterList;	
}

int calClassLabel(int queryBag, testDataSet &obj, float** housdroffDistMatrix, int R, int C)
{
	vector<int> voterList = calculateVoterList(queryBag, obj.bucketSizeArr.size(), housdroffDistMatrix, R, C);

	#ifdef debug
	cout<<"queryBag "<<queryBag<<" : ";
	for(int i=0; i<voterList.size(); i++)
		cout<<voterList[i]<<" ";
	cout<<endl;
	#endif //end of debug

	//calculating class label
	int classLabel = 0;

	for(int i=0; i<voterList.size(); i++)
	{
		if(obj.classLabel[voterList[i]] == 0)
			classLabel--;
		else
			classLabel++;
	}

	#ifdef debug
	cout<<"query bag - "<<queryBag<<" total classLabel value: "<<classLabel<<endl;
	#endif

	if(classLabel > 0)
	{
		classLabel = 1;
		//cout<<"MUSK\n";
	}
	else
	{
		classLabel = 0;
		//cout<<"NON_MUSK";
	}

	return classLabel;
}

int main()
{
	testDataSet obj;
	obj.readData(FILENAME);
	
	int R = R_VALUE;
	int C = R + 2;
	float correctPrediction = 0;
	int numOfBuckets = obj.bucketSizeArr.size();
	//int dataSize = obj.distMatrix.size();

	#ifdef debug
	cout<<"total rows: "<<(obj.distMatrix.size())/obj.columnSize<<endl;
	cout<<"total columns: "<<obj.columnSize<<endl;
	#endif

	float **housdroffDistMatrix = (float **) malloc(sizeof(float*) * numOfBuckets);
	for(int i=0; i<numOfBuckets; i++)
		housdroffDistMatrix[i] = (float *) malloc(sizeof(float) * numOfBuckets);
	
	calHousdroffDist(obj.distMatrix, obj.columnSize, obj.bucketLocArr, obj.bucketSizeArr, housdroffDistMatrix);

	#ifdef debug
	for(int i=0; i<numOfBuckets; i++)
	{
		for(int j=0; j<numOfBuckets; j++)
			cout<<housdroffDistMatrix[i][j]<<" ";
		
		cout<<endl;
	}
	#endif
	
	//loop to check accuracy of code
	for(int queryBag=0; queryBag<numOfBuckets; queryBag++)
	{
		int classLabel = calClassLabel(queryBag, obj, housdroffDistMatrix, R, C);
		if(classLabel == obj.classLabel[queryBag])
			correctPrediction++;
	}

	cout<<"correctPrediction = "<<correctPrediction<<endl;
	cout<<"dataSize = "<<numOfBuckets<<endl;
	float accuracy = (correctPrediction * 100.0)/(float) numOfBuckets;
	cout<<"accurary is "<<accuracy<<endl;

	return 0;
}