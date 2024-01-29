#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <random>

struct Particle {
    std::vector<double> position;
    std::vector<double> velocity;
    std::vector<double> best_position;
    double best_value;
};

double objective_function(double x, double y) {
    return std::pow(1.5 - x - x * y, 2) + std::pow(2.25 - x + x * y * y, 2) + std::pow(2.625 - x + x * y * y * y, 2);
}

void initialize_particles(std::vector<Particle>& particles, int dimensions, double range_min, double range_max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(range_min, range_max);

    for (Particle& p : particles) {
        p.position.resize(dimensions);
        p.velocity.resize(dimensions);
        p.best_position.resize(dimensions);

        for (int i = 0; i < dimensions; ++i) {
            p.position[i] = dis(gen);
            p.velocity[i] = dis(gen) / 2.0;
            p.best_position[i] = p.position[i];
        }

        p.best_value = objective_function(p.position[0], p.position[1]);
    }
}

void update_particles(std::vector<Particle>& particles, const std::vector<double>& global_best, double inertia, double cognitive, double social) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    for (Particle& p : particles) {
        for (size_t i = 0; i < p.position.size(); ++i) {
            double r1 = dis(gen);
            double r2 = dis(gen);

            // Update velocity
            p.velocity[i] = inertia * p.velocity[i] + cognitive * r1 * (p.best_position[i] - p.position[i]) + social * r2 * (global_best[i] - p.position[i]);

            // Update position
            p.position[i] += p.velocity[i];
        }

        double value = objective_function(p.position[0], p.position[1]);
        if (value < p.best_value) {
            p.best_value = value;
            p.best_position = p.position;
        }
    }
}

std::vector<double> particle_swarm_optimization(int dimensions, double range_min, double range_max, int num_particles, int max_iterations) {
    std::vector<Particle> particles(num_particles);
    initialize_particles(particles, dimensions, range_min, range_max);

    std::vector<double> global_best_position(dimensions);
    double global_best_value = std::numeric_limits<double>::infinity();

    for (int iter = 0; iter < max_iterations; ++iter) {
        for (const Particle& p : particles) {
            if (p.best_value < global_best_value) {
                global_best_value = p.best_value;
                global_best_position = p.best_position;
            }
        }

        update_particles(particles, global_best_position, 0.5, 2.0, 2.0);
    }

    return global_best_position;
}
