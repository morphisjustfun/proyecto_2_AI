#include "dataframe.h"

unsigned int Dataframe::get_nrows() const{
    return series.size();
}

unsigned int Dataframe::get_ncols() const{
    return headers.size();
}

std::vector<int> Dataframe::get_header_column(const std::string &header) const {
    auto find_value = std::find(headers.begin(), headers.end(), header);
    if (find_value == headers.end()) {
        throw std::invalid_argument("Header not found");
    }
    std::vector<int> values;
    unsigned int index = std::distance(headers.begin(), std::find(headers.begin(), headers.end(), header));
    for (auto &row: series) {
        values.push_back(row.at(index));
    }
    return values;
}


Dataframe Dataframe::read_csv(const std::string &filename, const char &delimiter) {
    std::ifstream file(filename);
    std::string line;
    std::vector<std::vector<int>> series;
    std::vector<std::string> headers;
    bool first_line = true;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string item;
        std::vector<int> row;
        while (std::getline(ss, item, delimiter)) {
            if (first_line) {
                headers.push_back(item);
            } else {
                row.push_back(std::stoi(item));
            }
        }
        if (!first_line) {
            series.push_back(row);
        }
        first_line = false;
    }
    return {series, headers};
}

std::vector<int> Dataframe::get_index_row(const unsigned int &index) const {
    if (index >= series.size()) {
        throw std::invalid_argument("Index out of range");
    }
    return series.at(index);
}
