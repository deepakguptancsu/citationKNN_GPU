#include "dataReader.h"

using namespace std;

void testDataSet::readData(string fileName)
{
	ifstream inputFile;
	inputFile.open(fileName.data());
	string data = "";
	int locVal = 0;

	//reading column size
	inputFile >> data;
	istringstream ss_temp(data);
	vector<string> rowData;
	string token;
	while(getline(ss_temp, token, ','))
		rowData.push_back(token);
	columnSize = rowData.size() - 3;
	//inputFile.close(fileName.data());

	//reading contents of input file into distance matrix
	//inputFile.open(fileName.data());
	inputFile.seekg(0, ios::beg);

	while(inputFile >> data)
	{
		istringstream ss(data);
		vector<string> rowData;
		//string token;
		bool foundBucketName = false;

		while(getline(ss, token, ','))
			rowData.push_back(token);
		
		for(int i=0; i<bucketName.size(); i++)
		{
			if(bucketName[i] == rowData[0])
			{
				foundBucketName = true;
				break;
			}
		}

		if(foundBucketName == false)
		{
			bucketName.push_back(rowData[0]);
			bucketLocArr.push_back(locVal);
			classLabel.push_back((rowData[rowData.size()-1])[0] - '0');
		}

		//vector<int> distFeatures;
		for(int i=2; i<rowData.size()-1; i++)
			distMatrix.push_back(stoi(rowData[i]));

		//distMatrix.push_back(distFeatures);
		locVal++; 
	}

	for(int i=0; i<bucketLocArr.size()-1; i++)
		bucketSizeArr.push_back(bucketLocArr[i+1] - bucketLocArr[i]);
	
	bucketSizeArr.push_back(((distMatrix.size())/columnSize) - bucketLocArr[bucketLocArr.size()-1]);
}

#if 0
void printData(testDataSet obj);

int main()
{
	testDataSet obj;
	obj.readData("my.data");
	//cout<<obj.bucketLocArr[obj.bucketLocArr.size()-1]<<endl;
	printData(obj);
	//printData();
	return 0;
}

void printData(testDataSet obj)
{
	cout<< "column size"<< obj.columnSize<<endl;
	cout<<"distance Matrix\n\n";

	for(int i=0; i<obj.distMatrix.size(); i++)
	{
		cout<<obj.distMatrix[i]<<" ";
		
		if((i+1) % obj.columnSize == 0)
			cout<<endl;
	}

	cout<<"bucket name\n\n";
	for(int i=0; i<obj.bucketName.size(); i++)
		cout<<obj.bucketName[i]<<" ";
	
	cout<<endl;

	cout<<"bucket loc arr\n\n";
	for(int i=0; i<obj.bucketLocArr.size(); i++)
		cout<<obj.bucketLocArr[i]<<" ";

	cout<<endl;

	cout<<"class label\n\n";
	for(int i=0; i<obj.classLabel.size(); i++)
		cout<<obj.classLabel[i]<<" ";
	
	cout<<endl;

	cout<<"bucket Size\n\n";
	for(int i=0; i<obj.bucketSizeArr.size(); i++)
		cout<<obj.bucketSizeArr[i]<<" ";
	
	cout<<endl;
	
}
#endif
