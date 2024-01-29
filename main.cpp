#include "RandomGraphs.hpp"
#include "Finding.hpp"
#include "PSO.hpp"

constexpr size_t cities = 12;

int main() {

	int dimensions = 2; // Two arguments for the function
	double range_min = -10;
	double range_max = 10;
	int num_particles = 30;
	int max_iterations = 100;

	std::vector<double> result = particle_swarm_optimization(dimensions, range_min, range_max, num_particles, max_iterations);
	std::cout << "Minimum found at: (" << result[0] << ", " << result[1] << ")" << std::endl;


	return 0;
	const auto g = create_random_graph(cities, false, true);
	// print_graph(g);

	//if (cities > 10) {
	//	throw std::exception("please don't");
	//}

	//BFS(g); // - needs (x - 1)! * (4 + x) bytes
	// DFS(g);
	//DFS2(g);
	// DFS2N(g);
	DFS3(g);

	NearestNeighbour(g);

	const Graph g_dijkstra = { // https://www.freecodecamp.org/news/dijkstras-shortest-path-algorithm-visual-introduction/
		// 0 1 2 3 4 5 6
		{0, 2, 6, INFINITY, INFINITY, INFINITY, INFINITY}, // 0
		{2, 0, INFINITY, 5, INFINITY, INFINITY, INFINITY}, // 1
		{6, INFINITY, 0, 8, INFINITY, INFINITY, INFINITY}, // 2
		{INFINITY, 5, 8, 0, 10, 15, INFINITY}, // 3
		{INFINITY, INFINITY, INFINITY, 10, 0, 6, 2}, // 4
		{INFINITY, INFINITY, INFINITY, 15, 6, 0, 6}, // 5
		{INFINITY, INFINITY, INFINITY, INFINITY, 2, 5, 0}, // 6
	};
	Dijkstra(g_dijkstra); 
}
