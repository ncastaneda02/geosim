#include <iostream>

#include <vector>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <glm/glm.hpp>
#include "photon_map.h"

void PhotonMap::buildMap(std::vector<Photon>* phots) {
	psize = phots->size();
	tree = new std::vector<Photon>(psize + 1);
	buildNode(phots, psize, 0, 0, 0);
}

void PhotonMap::printTree() {
	for (int i = 0; i < tree->size(); i++) {
		int axis = (int)trunc(log2(i)) % 4;
		std::cout << "(" << tree->at(i).position.x << ", " << tree->at(i).position.y << ", " << tree->at(i).position.z << ", " << tree->at(i).position.w << ")" << std::endl;
	}
}

void PhotonMap::buildNode(std::vector<Photon> *phots, size_t num_elems, size_t depth, size_t start, int ind) {
	if (num_elems <= 0) {
		return;
	}
	int axis = depth % 4;

	std::sort(phots->begin() + start, phots->begin() + start + num_elems,
		[axis](Photon& p1, Photon& p2) {
			return p1.position[axis] < p2.position[axis];
		});

	size_t mid = (num_elems - 1) / 2;
	Photon mid_phot = phots->at(start + mid);

	tree->at(ind) = mid_phot;
	buildNode(phots, mid, depth + 1, start, ind * 2 + 1);
	buildNode(phots, num_elems - mid - 1, depth + 1, start + mid + 1, ind * 2 + 2);
}

std::vector <PhotonMap::Photon> PhotonMap::kNearestNeighbors(int k, glm::dvec4 pos)
{
	std::vector<photon_pair> nearest_photons;
	kNearestHelper(k, 0, nearest_photons, pos, 0);
	std::vector<PhotonMap::Photon> nearby_photons;
	for (auto p = nearest_photons.begin(); p != nearest_photons.end(); p++)
	{
		nearby_photons.push_back(p->second);
	}
	return nearby_photons;
}

void PhotonMap::mapPush(std::vector<photon_pair>& nearest_photons, photon_pair p) {
	nearest_photons.push_back(p); std::push_heap(nearest_photons.begin(), nearest_photons.end(),
		[](photon_pair a, photon_pair b) -> bool
		{
			return a.first < b.first;
		});
}

void PhotonMap::mapPop(std::vector<photon_pair>& nearest_photons) {
	std::pop_heap(nearest_photons.begin(), nearest_photons.end(),
		[](photon_pair a, photon_pair b) -> bool
		{
			return  a.first < b.first;
		}); nearest_photons.pop_back();
}

void PhotonMap::kNearestHelper(int k, int depth, std::vector<photon_pair>& nearest_photons, glm::dvec4 pos, int ind)
{
	if (ind >= psize) {
		return;
	}

    auto tmp = tree->at(ind).position - pos;
	photon_pair med_pair = photon_pair((double)glm::dot(tmp, tmp), tree->at(ind));
	mapPush(nearest_photons, med_pair);
	if (nearest_photons.size() > k) {
		mapPop(nearest_photons);
	}


	// Recurse down the closer axis
	int axis = depth % 4;
	bool closer_left = pos[axis] < tree->at(ind).position[axis];
	if (closer_left)
	{
		kNearestHelper(k, depth + 1, nearest_photons, pos, ind * 2 + 1);
	}
	else {
		kNearestHelper(k, depth + 1, nearest_photons, pos, ind * 2 + 2);
	}

	// if it makes sense to search other side as well (minimum distance overlaps), do so
	double neighborDist = tree->at(ind).position[axis] - pos[axis];
	if (nearest_photons.front().first > neighborDist) {
		if (closer_left) {
			kNearestHelper(k, depth + 1, nearest_photons, pos, ind * 2 + 2);
		}
		else {
			kNearestHelper(k, depth + 1, nearest_photons, pos, ind * 2 + 1);
		}
	}
}