#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <string>
#include <random>
#include "sample.h"

namespace dataset
{
    std::vector<Sample> load_from_file(
        const std::string &path);

    struct DatasetSplit
    {
        std::vector<Sample> train;
        std::vector<Sample> test;
    };

    DatasetSplit split(
        const std::vector<Sample> &data,
        double train_ratio = 0.6, unsigned seed = 42);
}

#endif //DATASET_H