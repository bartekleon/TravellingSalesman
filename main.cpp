#include "RandomGraphs.hpp"
#include "Finding.hpp"

constexpr size_t cities = 5;

int main() {
	const auto g = create_random_graph(cities, false, false);
	// print_graph(g);

	//if (cities > 10) {
	//	throw std::exception("please don't");
	//}

	BFS(g);
	// DFS(g);
	//DFS2(g);
	// DFS2N(g);
	DFS3(g);
}
