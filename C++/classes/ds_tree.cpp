#include "ds_tree.h"

#include <utility>

DSTree::DSTree(const Dataframe &df, std::string label, split split_type) {
    this->construct_ds_tree(df, std::move(label), split_type);
}

void DSTree::construct_ds_tree(Dataframe df, std::string label_name, split split_type) {
    auto label_values = df.get_header_column(label_name);
    auto unique_label_values = count_unique(label_values);
    if (unique_label_values == 1) {
        this->value = 0;
        this->left = nullptr;
        this->right = nullptr;
        this->feature = "";
        this->leaf = true;
        this->label = label_values[0];
        return;
    }

    auto selected_feature = select_feature(df, label_name, split_type);
    auto selected_feature_mean = get_best_split(split_type)(df.get_header_column(selected_feature));

    // return 1 its right, 0 its left
    auto right_indexes = df.get_header_column(selected_feature) |
                         std::views::transform([selected_feature_mean](auto x) { return x > selected_feature_mean; });

    auto right_indexes_casted = std::vector<int>(right_indexes.begin(), right_indexes.end());

    std::vector<std::vector<int>> left_series;
    std::vector<std::vector<int>> right_series;

    for (int i = 0; i < right_indexes_casted.size(); i++) {
        if (right_indexes_casted[i] == 1) {
            right_series.push_back(df.series.at(i));
        } else {
            left_series.push_back(df.series.at(i));
        }
    }

    Dataframe left_df = Dataframe(left_series, df.headers);
    Dataframe right_df = Dataframe(right_series, df.headers);

    this->feature = selected_feature;
    this->value = selected_feature_mean;
    this->leaf = false;
    this->label = -1;

    if (!left_df.series.empty()) {
        this->left = new DSTree(left_df, label_name, split_type);
    }
    if (!right_df.series.empty()) {
        this->right = new DSTree(right_df, label_name, split_type);
    }
}

std::string DSTree::select_feature(const Dataframe &df, const std::string &label_name, split split_type) {
    auto label_values = df.get_header_column(label_name);
    auto features = df.headers | std::views::filter([&label_name](std::string value) { return value != label_name; });

    auto features_casted = std::vector<std::string>(features.begin(), features.end());

    std::vector<std::vector<int>> columns_categorical = std::vector<std::vector<int>>(df.get_ncols() - 1,
                                                                                      std::vector<int>(df.get_nrows(),
                                                                                                       0));

#pragma omp parallel for
    for (int i = 0; i < features_casted.size(); i++) {
        auto feature = features_casted[i];
        auto feature_values = df.get_header_column(feature);
        auto feature_split = get_best_split(split_type)(feature_values);

        auto feature_values_categorical = feature_values | std::views::transform([feature_split](int v) {
            return v > feature_split ? 1 : 0;
        });
        auto feature_values_categorical_casted = std::vector<int>(feature_values_categorical.begin(),
                                                                  feature_values_categorical.end());
        columns_categorical[i] = feature_values_categorical_casted;
    }

    std::vector<std::vector<int>> rows_categorical = std::vector<std::vector<int>>(columns_categorical[0].size(),
                                                                                   std::vector<int>(
                                                                                           columns_categorical.size(),
                                                                                           0));
#pragma omp parallel for
    for (int i = 0; i < columns_categorical[0].size(); i++) {
        std::vector<int> row = std::vector<int>(columns_categorical.size() + 1, 0);
        row[0] = label_values[i];
        for (int j = 0; j < columns_categorical.size(); j++) {
            row[j + 1] = columns_categorical[j][i];
        }
        rows_categorical[i] = row;
    }
    auto df_categorical = Dataframe(rows_categorical, df.headers);

    std::vector<double> feature_entropy = std::vector<double>(features_casted.size(), 0);
    std::vector<std::string> disordered_features = std::vector<std::string>(features_casted.size(), "");
#pragma omp parallel for
    for (int i = 0; i < features_casted.size(); i++) {
        auto feature = features_casted[i];

        auto label_values = get_unique<int>(df_categorical.get_header_column(label_name));
        auto total_data = df_categorical.get_nrows();

        auto entropy = get_entropy(feature, df_categorical, label_values, total_data, label_name);
        feature_entropy[i] = entropy;
        disordered_features[i] = feature;
    }
    auto min = std::min_element(feature_entropy.begin(), feature_entropy.end());
    auto min_index = std::distance(feature_entropy.begin(), min);
    return disordered_features[min_index];
}

double DSTree::get_entropy(const std::string &feature, const Dataframe &df_categorical, const std::vector<int>& label_values, unsigned int total_data, const std::string& label) {

    auto feature_yes_indexes =
            df_categorical.get_header_column(feature) | std::views::transform([](int v) { return v == 1 ? 1 : 0; });

    auto feature_yes_indexes_casted = std::vector<int>(feature_yes_indexes.begin(), feature_yes_indexes.end());

    std::vector<std::vector<int>> feature_yes_series;
    std::vector<std::vector<int>> feature_no_series;

    for (unsigned int i = 0; i < feature_yes_indexes_casted.size(); i++) {
        if (feature_yes_indexes_casted[i] == 1) {
            feature_yes_series.emplace_back(df_categorical.series.at(i));
        } else {
            feature_no_series.emplace_back(df_categorical.series.at(i));
        }
    }

    auto df_feature_yes = Dataframe(feature_yes_series, df_categorical.headers);
    auto df_feature_no = Dataframe(feature_no_series, df_categorical.headers);

    auto length_feature_yes = df_feature_yes.get_nrows();
    auto length_feature_no = df_feature_no.get_nrows();

    auto weight_feature_yes = double(length_feature_yes) / total_data;
    auto weight_feature_no = double(length_feature_no) / total_data;

    if (length_feature_yes == 0) {
        length_feature_yes = 1;
    }

    if (length_feature_no == 0) {
        length_feature_no = 1;
    }

    std::vector<double> entropy_feature_yes;
    std::vector<double> entropy_feature_no;
    double entropy_feature_yes_sum = 0;
    double entropy_feature_no_sum = 0;

#pragma omp parallel for reduction(+:entropy_feature_yes_sum) reduction(+:entropy_feature_no_sum)
    for (int i = 0; i < label_values.size(); i++) {
        auto label_value = label_values[i];
        auto label_yes = df_feature_yes.get_header_column(label) | std::views::filter(
                [label_value](int x) { return x == label_value; });
        auto label_no = df_feature_no.get_header_column(label) | std::views::filter(
                [label_value](int x) { return x == label_value; });

        auto label_yes_casted = std::vector<int>(label_yes.begin(), label_yes.end());
        auto label_no_casted = std::vector<int>(label_no.begin(), label_no.end());

        auto label_yes_prob = double(label_yes_casted.size()) / length_feature_yes;
        auto label_no_prob = double(label_no_casted.size()) / length_feature_no;

        if (label_yes_prob == 0) {
            label_yes_prob = 1;
        }

        if (label_no_prob == 0) {
            label_no_prob = 1;
        }

        entropy_feature_yes_sum += -label_yes_prob * log2(label_yes_prob);
        entropy_feature_no_sum += -label_no_prob * log2(label_no_prob);
    }
    return weight_feature_yes * entropy_feature_yes_sum + weight_feature_no * entropy_feature_no_sum;
}

std::vector<bool> DSTree::eval_df(DSTree ds_tree, Dataframe df, std::string label_name) {
    auto label_index = std::find(df.headers.begin(), df.headers.end(), label_name) - df.headers.begin();
    std::vector<bool> results;
    for (auto const &serie: df.series) {
        results.emplace_back(ds_tree.eval_serie(serie, label_index, df.headers));
    }
    return results;
}

bool DSTree::eval_serie(std::vector<int> serie, unsigned int label_index, std::vector<std::string> headers) const {
    if (this->leaf) {
        return this->label == serie[label_index];
    }

    auto feature_index = std::find(headers.begin(), headers.end(), this->feature) - headers.begin();
    if (serie[feature_index] > this->value) {
        return this->right->eval_serie(serie, label_index, headers);
    }
    return this->left->eval_serie(serie, label_index, headers);
}

