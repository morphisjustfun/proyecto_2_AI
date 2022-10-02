#pragma once

#include <algorithm>
#include <fstream>
#include <sstream>
#include <numeric>
#include <functional>

enum split {
    mean,
    median,
    mode
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

std::function<double(std::vector<int>)> get_best_split(split type);

