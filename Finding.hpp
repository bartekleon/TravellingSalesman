#pragma once

#include <map>
#include <random>
#include <cmath>
#include <iostream>
#include <set>
#include <queue>
#include <string_view>
#include <utility>
#include <algorithm>
#include <execution>
#include <bitset>

typedef std::vector<std::vector<float>> Graph;

void BFS(const Graph& graph) {
    std::vector<std::pair<std::string, float>> current;
    std::vector<std::pair<std::string, float>> current_2;

    current.push_back(std::make_pair("0", 0));

    const size_t size = graph.size();

    for (size_t i = 1; i < size; ++i) {
        for (size_t j = 0; j < current.size(); ++j) {
            const auto& pair = current[j];
            for (size_t k = 0; k < size; ++k) {
                bool found = false;
                for (const char ch : pair.first) {
                    if (ch == k + 48) {
                        found = true;
                        break;
                    }
                }
                if (found) continue;

                const char last_char = pair.first.at(pair.first.size() - 1) - 48;
                const auto f = graph.at(last_char).at(k);

                current_2.push_back(
                    std::make_pair(pair.first + (char)(k + 48), pair.second + f)
                );
            }
        }
        std::swap(current, current_2);
        current_2.clear();
    }

    for (auto& c : current) {
        auto f = c.first[c.first.size() - 1] - 48;
        c.second += graph.at(f).at(0);
    }

    std::sort(
        current.begin(),
        current.end(),
        [](std::pair<std::string, float> a, std::pair<std::string, float> b) {
            return a.second < b.second;
        }
    );

    for (size_t i = 0; i < std::min(current.size(), current.size()); ++i) {
        std::cout << "known result: " << current[i].first << " " << current[i].second << std::endl;
    }
}

float calculate_weight(const Graph& graph, const std::vector<char>& answer) {
    float weight = 0;
    for (size_t i = 0; i < answer.size() - 1; ++i) {
        weight += graph[answer[i]][answer[i + 1]];
    }
    weight += graph[answer[answer.size() - 1]][0];

    return weight;
}

float _DFS(size_t graph_size, uint64_t used, const Graph& graph, std::vector<char>& answer) {
    if (answer.size() == graph_size) {
        return calculate_weight(graph, answer);
    }
    float min = INFINITY;

    for (size_t i = 1; i < graph_size; ++i) {
        if (used & (1ull << i)) {
            continue;
        }
        answer.emplace_back(i);
        min = std::min(min, _DFS(graph_size, used ^ (1ull << i), graph, answer));
        answer.pop_back();
    }

    return min;
}

void DFS(const Graph& graph) {
    std::vector<char> answer{ 0 };
    float min = _DFS(graph.size(), 0, graph, answer);

    std::cout << "res dfs1: " << min << std::endl;
}

void DFS2(const Graph& graph) {
    const size_t graph_size = graph.size();
    std::mutex min_values_mutex;

    std::vector<float> min_values(graph_size, INFINITY);
    std::vector<char> starting_pos(graph_size - 1, 0);

    for (char i = 0; i < graph_size; ++i) {
        starting_pos[i] = i + 1;
    }

    // Parallelize the for loop using C++17 Parallel STL
    std::for_each(std::execution::par, starting_pos.begin(), starting_pos.end(), [&](char value) {
        std::vector<char> answer{ 0, value };
        float v = _DFS(graph_size, 1ull << value, graph, answer);
        std::lock_guard<std::mutex> lock(min_values_mutex);
        min_values[value] = v;
    });

    std::cout << "res dfs2: " << *std::min_element(min_values.begin() + 1, min_values.end()) << std::endl << std::endl;
}

float _DFS2(size_t graph_size, uint64_t used, float result, const Graph& graph, std::vector<char>& answer) {
    if (answer.size() == graph_size) {
        return result + graph[answer[answer.size() - 1]][0];
    }
    float min = INFINITY;

    for (size_t i = 1; i < graph_size; ++i) {
        if (used & (1ull << i)) {
            continue;
        }
        float val = graph[answer[answer.size() - 1]][i];
        answer.emplace_back(i);
        used ^= 1ull << i;
        float nv = _DFS2(graph_size, used, result + val, graph, answer);
        min = std::min(min, nv);
        used ^= 1ull << i;
        answer.pop_back();
    }

    return min;
}

void DFS2N(const Graph& graph) {
    const size_t graph_size = graph.size();

    std::vector<float> min_values(graph_size, INFINITY);
    std::vector<char> starting_pos(graph_size - 1, 0);
    std::mutex min_values_mutex;

    for (char i = 0; i < graph_size - 1; ++i) {
        starting_pos[i] = i + 1;
    }

    std::for_each(std::execution::par, starting_pos.begin(), starting_pos.end(), [&](char value) {
        std::vector<char> answer{ 0, value };

        const float v = _DFS2(graph_size, 1ull << value, graph[0][value], graph, answer);
        std::lock_guard<std::mutex> lock(min_values_mutex);
        min_values[value] = v;
    });

    std::cout << "res: " << *std::min_element(min_values.begin(), min_values.end()) << std::endl << std::endl;
}

size_t get_size(uint64_t used) {
    return 16ull - (std::countl_zero(used) >> 2);
}

size_t get_nth(uint64_t used, uint64_t pos) {
    return (used >> (pos << 2u)) & 0xF;
}

struct _DFS3Result {
    uint64_t answer;
    float value;
};

bool operator<(const _DFS3Result& a, const _DFS3Result& b) {
    return a.value < b.value;
}

_DFS3Result _DFS3(size_t graph_size, uint64_t used, float result, uint64_t answer, const Graph& graph) {
    const auto size = get_size(answer);
    if ((size + 1) == graph_size) {
        return { answer, result + graph[get_nth(answer, size - 1)][0] };
    }
    _DFS3Result min{ 0 , INFINITY };

    for (size_t i = 1; i < graph_size; ++i) {
        if (used & (1ull << i)) {
            continue;
        }
        const auto new_used = used ^ (1ull << i);
        const auto new_answer = answer ^ (i * (1ull << (size * 4)));
        const auto new_result = result + graph[get_nth(answer, size - 1)][i];
        const auto val = _DFS3(graph_size, new_used, new_result, new_answer, graph);
        if (min.value > val.value) {
            min = val;
        }
    }

    return min;
}

void DFS3(const Graph& graph) {
    const size_t graph_size = graph.size();

    std::vector<_DFS3Result> min_values(graph_size);
    std::vector<char> starting_pos(graph_size - 1, 0);
    std::mutex min_values_mutex;

    for (char i = 0; i < graph_size - 1; ++i) {
        starting_pos[i] = i + 1;
    }

    std::for_each(std::execution::par, starting_pos.begin(), starting_pos.end(), [&](char value) {
        const _DFS3Result v = _DFS3(graph_size, 1ull << value, graph[0][value], (size_t)value, graph);
        std::lock_guard<std::mutex> lock(min_values_mutex);
        min_values[value] = v;
    });

    const auto& res = *std::min_element(min_values.begin() + 1, min_values.end());

    std::cout << "res: " << res.answer << " " << res.value << std::endl << std::endl;
}

//A*, Ant colonial optimisation, Nearest neighbour, Other greedy

void NearestNeighbour(const Graph& graph) {
    const size_t graph_size = graph.size();
}

void AStar(const Graph& graph) {
    const size_t graph_size = graph.size();
}
