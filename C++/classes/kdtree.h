#include <array>
#include <vector>
#include <algorithm>
#include <queue>
#include <memory>
#include <limits>
#include <map>
#include "../utils/cfunctions.h"
#include "dataframe.h"

class KDTree {
private:
    using Point = std::vector<float>;

    struct Node {
        std::vector<float> point;
        std::shared_ptr<Node> left;
        std::shared_ptr<Node> right;
        int label;
        bool infinite = false;

        double get_point_distance(const Point &other) const {
            if (infinite)
                return std::numeric_limits<double>::infinity();
            return get_distance<float>(point, other, euclidean);
        }
    };

    using NodePtr = std::shared_ptr<Node>;
    using NodePQ = std::vector<NodePtr>;


    NodePtr root;
    size_t dim;

    void insert(std::vector<std::pair<Point, int>> &points, const size_t from, const size_t to,
                NodePtr &current,
                const size_t depth) {
        if (from == to) return;
        const size_t axis = depth % this->dim;
        const size_t median = (from + to) / 2;
        std::sort(points.begin() + from, points.begin() + to,
                  [axis](std::pair<Point, int> const &a,
                         std::pair<Point, int> const &b) { return a.first[axis] < b.first[axis]; });
        current = std::make_shared<Node>(points[median].first, nullptr, nullptr, points[median].second, false);
        insert(points, from, median, current->left, depth + 1);
        insert(points, median + 1, to, current->right, depth + 1);
    }

public:
    explicit KDTree(Dataframe df, std::string label) {
        auto data = df.series;
        auto label_index = std::find(df.headers.begin(), df.headers.end(), label) - df.headers.begin();

        std::vector<std::pair<Point, int>> data_processed;
        for (auto row: data) {
            Point point;
            for (size_t i = 0; i < row.size(); i++) {
                if (i != label_index) {
                    point.push_back(row[i]);
                }
            }
            data_processed.emplace_back(point, row[label_index]);
        }
        this->dim = data_processed[0].first.size();
        insert(data_processed, 0, data_processed.size(), root, 0);
    }

    static std::vector<int> eval_df(const KDTree &root, Dataframe df, const std::string &label, const size_t &k) {
        auto data = df.series;
        auto label_index = std::find(df.headers.begin(), df.headers.end(), label) - df.headers.begin();

        std::vector<int> results;
        for (auto row: data) {
            Point point;
            for (size_t i = 0; i < row.size(); i++) {
                if (i != label_index) {
                    point.push_back(row[i]);
                }
            }
            auto knn = root.knn(point, k);
            // get the most common label from knn
            std::map<int, int> label_count;
            for (auto &node: knn) {
                if (label_count.find(node->label) == label_count.end()) {
                    label_count[node->label] = 1;
                } else {
                    label_count[node->label]++;
                }
            }
            std::sort(knn.begin(), knn.end(),
                      [label_count](NodePtr const &a, NodePtr const &b) {
                          return label_count.at(a->label) > label_count.at(b->label);
                      });
            results.push_back(knn[0]->label);
        }
        return results;
    }

    bool contains(Point const &point) {
        size_t depth = 0;
        auto current = root;
        while (current) {
            if (is_vector_equal(point, current->point)) return true;
            if (point[depth % this->dim] < current->point[depth % this->dim]) current = current->left;
            else current = current->right;
            depth += 1;
        }
        return false;
    }

    NodePQ knn(const Point &point, size_t k) const {
        NodePQ pq;
        auto compare_distance = [point](const std::pair<double, NodePtr> &a, const std::pair<double, NodePtr> &b) {
            return a.second->get_point_distance(point) > b.second->get_point_distance(point);
        };

        std::priority_queue<std::pair<double, NodePtr>, std::vector<std::pair<double, NodePtr>>, decltype(compare_distance)> queue(
                compare_distance);
        auto current = root;
        size_t depth = 0;
        while (current) {
            queue.push(std::make_pair(current->get_point_distance(point), current));
            if (point[depth % this->dim] < current->point[depth % this->dim]) current = current->left;
            else current = current->right;
            depth += 1;
        }

        while (k > 0) {
            auto pair = queue.top();
            queue.pop();
            pq.push_back(pair.second);
            k -= 1;
        }

        return pq;
    }
};
