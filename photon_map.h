#pragma once

#include <iostream>

#include <vector>
#include <glm/glm.hpp>


// Spatially sorted map of photons. Basically just a kD tree
class PhotonMap
{
	public:
		struct Photon
		{
			glm::dvec4 position;
			glm::dvec4 radiance;
			glm::dvec4 wi;

			Photon() {}
			Photon(glm::dvec4 pos, glm::dvec4 rad, glm::dvec4 dir) : position(pos), radiance(rad), wi(dir) {}

		};


		PhotonMap() {}

		~PhotonMap()
		{
			delete root;
		}

		void buildMap(std::vector<Photon> *photons);
		std::vector <PhotonMap::Photon> kNearestNeighbors(int k, glm::dvec4 pos);

		int size() { return psize; }


	private:

		struct PhotonNode
		{
			Photon photon;
			PhotonNode* left, *right;
			double left_dist;
			double right_dist;

			~PhotonNode() {
				if (left) delete left;
				if (right) delete right;
			}
		};

		using photon_pair = std::pair<double, PhotonMap::Photon>;

		PhotonNode* root;
		std::vector<Photon> *tree;
		
		int psize;

		void buildNode(std::vector<Photon>* phots, size_t num_elems, size_t depth, size_t start, int ind);
		void printTree();

		void mapPush(std::vector<photon_pair>& nearest_photons, photon_pair p);
		void mapPop(std::vector<photon_pair>& nearest_photons);

		void kNearestHelper(int k, int depth, std::vector<photon_pair>& nearest_photons, glm::dvec4 pos, int ind);
};
