#include <iostream>

#include <vector>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <glm/glm.hpp>
#include "photon_map.h"

void PhotonMap::buildMap(std::vector<Photon>* phots) {
	photons = phots;

	root = buildNode(phots, phots->size(), 0, 0);
}

PhotonMap::PhotonNode *PhotonMap::buildNode(std::vector<Photon> *phots, size_t num_elems, size_t depth, size_t start) {
	if (num_elems <= 0) {
		return nullptr;
	}
	int axis = depth % 3;

	std::sort(phots->begin() + start, phots->begin() + start + num_elems,
		[axis](Photon& p1, Photon& p2) {
			return p1.position[axis] < p2.position[axis];
		});

	size_t mid = (num_elems - 1) / 2;
	Photon mid_phot = phots->at(start + mid);

	PhotonNode *new_node = new PhotonNode();
	new_node->photon = mid_phot;
	new_node->left = buildNode(phots, mid, depth + 1, start);
	new_node->right = buildNode(phots, num_elems - mid - 1, depth + 1, start + mid + 1);
	return new_node;

}

std::vector <PhotonMap::Photon> PhotonMap::kNearestNeighbors(int k, glm::vec4 pos)
{
	std::vector<photon_pair> nearest_photons;
	kNearestHelper(root, k, 0, nearest_photons, pos);
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

void PhotonMap::kNearestHelper(PhotonNode* start, int k, int depth, std::vector<photon_pair>& nearest_photons, glm::vec4 pos)
{
	if (start == nullptr) {
		return;
	}

    auto tmp = start->photon.position - pos;
	photon_pair med_pair = photon_pair((double)glm::dot(tmp, tmp), start->photon);
	mapPush(nearest_photons, med_pair);
	if (nearest_photons.size() > k) {
		mapPop(nearest_photons);
	}


	// Recurse down the closer axis
	int axis = depth % 3;
	bool closer_left = pos[axis] < start->photon.position[axis];
	if (closer_left)
	{
		kNearestHelper(start->left, k, depth + 1, nearest_photons, pos);
	}
	else {
		kNearestHelper(start->right, k, depth + 1, nearest_photons, pos);
	}

	// if it makes sense to search other side as well (minimum distance overlaps), do so
	double neighborDist = start->photon.position[axis] - pos[axis];
	if (nearest_photons.front().first > neighborDist * neighborDist * neighborDist) {
		if (closer_left) {
			kNearestHelper(start->right, k, depth + 1, nearest_photons, pos);
		}
		else {
			kNearestHelper(start->left, k, depth + 1, nearest_photons, pos);
		}
	}
}