#pragma once

#include <algorithm>
#include <fstream>
#include <sstream>
#include <numeric>
#include <functional>
#include <cmath>

enum split {
    mean,
    median,
    mode
};

enum distanceType {
    euclidean,
    manhattan,
    minkowski
};


template<typename T>
unsigned int count_unique(std::vector<T> vec) {
    std::sort(vec.begin(), vec.end());
    return std::unique(vec.begin(), vec.end()) - vec.begin();
}

template<typename T>
std::vector<T> get_unique(std::vector<T> vec) {
    std::sort(vec.begin(), vec.end());
    vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
    return vec;
}

template<typename T>
bool is_vector_equal(std::vector<T> const &a, std::vector<T> const &b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}

std::function<double(std::vector<int>)> get_best_split(split type);

template<typename T>
double get_distance(std::vector<T> first, std::vector<T> second, distanceType type) {
    if (first.size() != second.size()) {
        throw std::invalid_argument("Vectors must be of the same size");
    }
    double distance = 0;
    switch (type) {
        case euclidean:
            for (int i = 0; i < first.size(); i++) {
                distance += pow(first[i] - second[i], 2);
            }
            return sqrt(distance);
        case manhattan:
            for (int i = 0; i < first.size(); i++) {
                distance += abs(first[i] - second[i]);
            }
            return distance;
        case minkowski:
            for (int i = 0; i < first.size(); i++) {
                distance += pow(abs(first[i] - second[i]), 2);
            }
            return std::pow(distance, 1.0 / 2.0);
        default:
            throw std::invalid_argument("Invalid distance type");
    }
}