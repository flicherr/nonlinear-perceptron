#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include "dataset.h"

std::vector<Sample> dataset::load_from_file(
    const std::string &path)
{
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Cannot open dataset file");
    }

    std::vector<Sample> data;
    std::string line;

    if (!std::getline(file, line)) {
        throw std::runtime_error("Empty CSV file");
    }

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        Sample s{};

        if (!std::getline(ss, token, ','))
            continue;
        s.x = std::stod(token);

        if (!std::getline(ss, token, ','))
            continue;
        s.y = std::stod(token);

        data.push_back(s);
    }

    return data;
}

dataset::DatasetSplit dataset::split(
        const std::vector<Sample> &data,
        double train_ratio, unsigned seed)
{
    std::vector<Sample> shuffled = data;

    std::mt19937 rng(seed);
    std::ranges::shuffle(shuffled, rng);

    auto train_size =
        static_cast<int>(train_ratio * shuffled.size());

    DatasetSplit split;
    split.train.assign(
        shuffled.begin(), shuffled.begin() + train_size);
    split.test.assign(
        shuffled.begin() + train_size, shuffled.end());

    return split;
}