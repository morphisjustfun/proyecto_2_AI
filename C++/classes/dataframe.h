#pragma once
#include <algorithm>
#include <utility>
#include <vector>
#include <string>
#include <stdexcept>
#include <any>
#include <iostream>
#include <fstream>
#include <sstream>


class Dataframe {

public:
    std::vector<std::vector<int>> series;
    std::vector<std::string> headers;
    [[nodiscard]] unsigned int get_nrows() const;

    [[nodiscard]] unsigned int get_ncols() const;

    Dataframe(std::vector<std::vector<int>> series, std::vector<std::string> headers) {
        this->series = std::move(series);
        this->headers = std::move(headers);
    }

    Dataframe(const Dataframe& other) {
        this->series = other.series;
        this->headers = other.headers;
    }

    [[nodiscard]] std::vector<int> get_header_column(const std::string& header) const;

    [[nodiscard]] std::vector<int> get_index_row(const unsigned int& index) const;

    static Dataframe read_csv(const std::string& filename, const char& delimiter = ',');
};
