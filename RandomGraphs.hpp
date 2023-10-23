#pragma once

#include <map>
#include <random>
#include <cmath>
#include <iostream>
#include <vector>
#include <ranges>

typedef std::vector<std::vector<float>> Graph;

float distance(float x1, float y1, float x2, float y2) {
	return std::hypot(x1 - x2, y1 - y2);
}

Graph create_random_graph(size_t size, bool is_symetric = true, bool is_full = true) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dist(1.0, 10.0);
	std::uniform_real_distribution<float> dist_symm(0.0, 1.0);
	std::uniform_real_distribution<float> dist_full(0.0, 1.0);

	std::vector<std::pair<float, float>> points;

	for (size_t i = 0; i < size; ++i) {
		points.push_back({ dist(gen), dist(gen) });
	}

	Graph graph;

	for (uint32_t i = 0; i < size; ++i) {
		graph.push_back(std::vector<float>(size, 0));
	}

	// Create edges and calculate weights
	for (uint32_t i = 0; i < size; ++i) {
		for (uint32_t j = i + 1; j < size; ++j) {
			const float weight = distance(points[i].first, points[i].second, points[j].first, points[j].second);
			const auto exists = dist_full(gen) >= 0.8;

			if (!is_full && dist_full(gen) < 0.2) {
				graph[i][j] = INFINITY;
				graph[j][i] = INFINITY;
				continue;
			}

			if (is_symetric) {
				graph[i][j] = weight;
				graph[j][i] = weight; // Symmetric graph
			}
			else {
				const auto amplitude = dist_symm(gen) > 0.5;
				if (amplitude) {
					graph[i][j] = weight * 1.2;
					graph[j][i] = weight;
				}
				else {
					graph[i][j] = weight;
					graph[j][i] = weight * 1.2;
				}
			}
		}
	}

	return graph;
}

void print_graph(const Graph& graph) {
	// Print the generated graph
	for (const auto& [idx, vertex] : std::views::enumerate(graph)) {
		std::cout << "V " << idx << " --: ";
		for (const auto& [idx2, neighbor] : std::views::enumerate(vertex)) {
			std::cout << idx2 << " (" << neighbor << "), ";
		}
		std::cout << std::endl;
	}
}