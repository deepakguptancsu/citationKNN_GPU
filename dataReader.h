#ifndef DATAREADER_H
#define DATAREADER_H

#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>
#include<string>

using namespace std;

class testDataSet
{
	public:
	vector<int> distMatrix;
	int columnSize;
	//int *distMatrix;
	vector<string> bucketName;
	vector<int> bucketLocArr;
	vector<int> bucketSizeArr;
	vector<int> classLabel;
	void readData(string fileName);
};

#endif