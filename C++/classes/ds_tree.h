#pragma once

#include <string>
#include "ranges"
#include <iostream>
#include <cmath>
#include <memory>
#include "../utils/cfunctions.h"
#include "dataframe.h"
#include "omp.h"


class DSTree {
    std::string feature;
    double value{};
    bool leaf{};
    int label;
    std::shared_ptr<DSTree> left;
    std::shared_ptr<DSTree> right;

public:
    explicit DSTree(const Dataframe &df, std::string label, split split_type);

    void construct_ds_tree(Dataframe df, std::string label_name, split split_type);

    static std::vector<int> eval_df(DSTree ds_tree, Dataframe df, std::string label_name);

private:
    static std::string select_feature(const Dataframe &df, const std::string &label_name, split split_type);

    static double
    get_entropy(const std::string &feature, const Dataframe &df_categorical, const std::vector<int> &label_values,
                unsigned int total_data, const std::string &label);

    [[nodiscard]] int
    eval_serie(std::vector<int> serie, unsigned int label_index, std::vector<std::string> headers) const;
};
