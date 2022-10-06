#include <iostream>
#include "classes/ds_tree.h"
#include "classes/kdtree.h"
#include <chrono>

int main() {
    auto df_train = Dataframe::read_csv("./data/train/test.csv");
    auto df_test = Dataframe::read_csv("./data/test/sign_mnist_test.csv");
//    auto begin = std::chrono::high_resolution_clock::now();
//    auto test = DSTree(df_train, "label", median);
//    auto end = std::chrono::high_resolution_clock::now();
//    auto test_results = DSTree::eval_df(test, df_test, "label");
//
//    int correct = 0;
//    for (auto const &result: test_results) {
//        if (result) correct++;
//    }
//    std::cout << "Correct: " << correct << std::endl;
//    std::cout << "Total: " << test_results.size() << std::endl;
//    std::cout << "Accuracy: " << double(correct) / test_results.size() << std::endl;
//    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;

    auto begin = std::chrono::high_resolution_clock::now();
    auto test = KDTree<float, int>(df_train, "label");
    auto end = std::chrono::high_resolution_clock::now();
    auto test_results = KDTree<float,int>::eval_df(test, df_test, "label", 4);

    int correct = 0;
    for (auto const &result: test_results) {
        if (result) correct++;
    }
    std::cout << "Correct: " << correct << std::endl;
    std::cout << "Total: " << test_results.size() << std::endl;
    std::cout << "Accuracy: " << double(correct) / test_results.size() << std::endl;
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;

    return 0;
}