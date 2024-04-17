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
			glm::vec4 position;
			glm::vec4 radiance;
			glm::vec4 wi;

			Photon() {}
			Photon(glm::vec4 pos, glm::vec4 rad, glm::vec4 dir) : position(pos), radiance(rad), wi(dir) {}

		};


		PhotonMap() {}

		~PhotonMap()
		{
			delete root;
			delete photons;
		}

		void buildMap(std::vector<Photon> *photons);
		std::vector <PhotonMap::Photon> kNearestNeighbors(int k, glm::vec4 pos);

		int size() { return photons->size(); }


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

		std::vector<Photon> *photons;
		PhotonNode* root;
		std::vector<bool> inUpperHalf;

		PhotonMap::PhotonNode* buildNode(std::vector<Photon>* phots, size_t num_elems, size_t depth, size_t start);

		void mapPush(std::vector<photon_pair>& nearest_photons, photon_pair p);
		void mapPop(std::vector<photon_pair>& nearest_photons);


		void kNearestHelper(PhotonNode* start, int k, int depth, std::vector<photon_pair>& nearest_photons, glm::vec4 pos);
		bool leftCloser(PhotonNode* node, glm::vec4& pos);
};
