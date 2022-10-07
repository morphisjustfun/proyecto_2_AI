#include <iostream>
#include "classes/ds_tree.h"
#include "classes/kdtree.h"


template<typename T>
void vector_to_csv(const std::vector<T>& target, const std::string& header, const std::string& filename) {
    std::ofstream file;
    file.open(filename);
    file << header << std::endl;
    for (auto& i : target) {
        file << i << std::endl;
    }
    file.close();
}

int main(int argc, char *argv[]) {
    auto df_train = Dataframe::read_csv("../Python/data/train.csv");
    auto df_test = Dataframe::read_csv("../Python/data/test.csv");
    auto type_str = std::string(argv[1]);
    if (type_str == "kdtree") {
        KDTree tree(df_train, "label");
        auto k_str = std::string(argv[2]);
        auto k_int = std::stoi(k_str);
        auto test_results = KDTree::eval_df(tree, df_test, "label", k_int);
        vector_to_csv(test_results, "label", "../Python/data/kdtree_results.csv");
    } else if (type_str == "ds_tree") {
        auto partition_str = std::string(argv[2]);
        auto split = split::mean;
        if (partition_str == "mean") {
            split = split::mean;
        } else if (partition_str == "median") {
            split = split::median;
        } else if (partition_str == "mode") {
            split = split::mode;
        }

        DSTree tree(df_train, "label", split);
        auto test_results = DSTree::eval_df(tree, df_test, "label");
        vector_to_csv(test_results, "label", "../Python/data/ds_tree_results.csv");
    }
    return 0;
}