#pragma once
#include <vector>
#include <algorithm>
#include <map>
#include <iostream>
#include <cmath>
#include "dataset.cpp"

class Knn {

public:
	Knn() = default;
	void Fit(Dataset &data);
	int Predict(std::vector<int> val, int k);
	std::vector<std::pair<std::vector<float>,int>> Neighbors;



private:
	Dataset Data;
	float euclideanDistance(std::vector<int> x, std::vector<int>  y);


};

//find the actual label by extracting the label from the map with biggest value.
int find_max(std::map<int, int> counts)
{
	std::map<int, int>::iterator itr;
	int max = -1;
	int lbl = 0;
	for (itr = counts.begin(); itr != counts.end(); ++itr) {
		if (itr->second > max)
		{
			max = itr->second;
			lbl = itr->first;
		}
	}
	return lbl;
}

void  Knn::Fit(Dataset &data)
{
	this->Data = data;
}
int  Knn::Predict(std::vector<int> val,int k)
{
	/*
	 * Neighbors: Holds the closest k points after calling prediction
	 * counts:Keeps the label,labelCount pair. is a map
	 * Points: Keeps all points as a float,vector(float) pair. First one is distance
	 * of our value to the 'point' in the dataset, which is also second parameter.
	 * distances: vector of float,int pair which holds distance and label values.
	 * it is used to keep integration with the Points vector.
	 */
	float min = RAND_MAX;
	std::vector<std::pair<float, std::vector<int>>> Points;
	Neighbors.clear();
	std::vector<std::pair<float, int>> distances;

	for (size_t i = 0; i < Data.GetData().size(); ++i)
	{
		std::vector<int>  currentRSSI = {Data.GetData().at(i).RSSI_max, Data.GetData().at(i).RSSI_min};
		float dist = euclideanDistance(val, currentRSSI);
		int lbl= Data.GetLabels().at(i);

		distances.push_back(std::pair<float, int>(dist,lbl));
		Points.push_back(std::pair<float, std::vector<int>>(dist, currentRSSI));

	}
	//Sort vectors by distance.
	std::sort(distances.begin(), distances.end());
	std::sort(Points.begin(), Points.end());

	std::map<int,int> counts;

	for (int i = 0; i < k; i++)
	{
		int lbl = distances.at(i).second;
		Neighbors.push_back(std::pair<std::vector<float>, int>(Points.at(i).second,lbl));

		if (!counts.count(lbl))
			counts.insert(std::pair<int, int>(lbl, 1));
		else
			counts[lbl]++;
	}
	return find_max(counts);
	

}

//our distance function
//TODO:Add different distace functions like Manhattan distance. Each distance function has a specific use.
float  Knn::euclideanDistance(std::vector<int>  x, std::vector<int>  y) {
	return  std::sqrt(std::pow(x.at(0)-y.at(0),2)+ std::pow(x.at(1) - y.at(1), 2));
}

int main(std::vector<int> value) { //Receive the position

	Dataset data = Dataset();
	data.Print();

	
	Knn knn = Knn();
	knn.Fit(data);
	int prediction=knn.Predict(value,7);

	for (size_t i = 0; i < knn.Neighbors.size(); i++)
	{
		std::cout<<knn.Neighbors.at(i).first.at(0)<<"   " << knn.Neighbors.at(i).first.at(1) <<std::endl;
		std::cout << knn.Neighbors.at(i).second << std::endl;
	}
	std::cout << "Labl:"<<prediction;

	
	
}
