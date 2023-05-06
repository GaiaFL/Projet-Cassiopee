#pragma once
#ifndef BLOGS_DATASET_H
#define BLOGS_DATASET_H
#include <ctime>

#endif //BLOGS_DATASET_H
#include <iostream>
#include <vector>

struct Local{
    char MAC[12];
    int Delta;
    int RSSI_min;
    int RSSI_max;
};

class Dataset {

public:
	//Create dataset
	Dataset(){
        
    };

	std::vector<Local> GetData();
	std::vector<int> GetLabels();
	void Print();
private:
	std::vector<Local> Data;
	std::vector<int> Labels;

};

std::vector<Local> Dataset::GetData()
{
	return Data;
}
std::vector<int> Dataset::GetLabels()
{
	return Labels;
}
void Dataset::Print()
{
	for (size_t i = 0; i < Data.size(); i++) {
		std::cout << "MAC Address:" << Data.at(i).MAC << +" Delta:" << Data.at(i).Delta << +" RSSIm:" << Data.at(i).RSSI_min +" RSSIM:" << Data.at(i).RSSI_max << " " << Labels.at(i) << std::endl;
	}
}