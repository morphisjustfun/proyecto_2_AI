#include "cfunctions.h"


std::function<double(std::vector<int>)> get_best_split(split type) {
    return [type](std::vector<int> values) {
        double result;
        switch (type) {
            case mean:
                result = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
                break;
            case median:
                std::sort(values.begin(), values.end());
                if (values.size() % 2 == 0) {
                    result = (values.at(values.size() / 2 - 1) + values.at(values.size() / 2)) / 2.0;
                } else {
                    result = values.at(values.size() / 2);
                }
                break;
            case mode:
                std::sort(values.begin(), values.end());
                result = values[0];
                break;
        }
        return result;
    };
}

