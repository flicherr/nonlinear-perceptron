#ifndef TRAINER_H
#define TRAINER_H

#include "network/network.h"
#include "data/sample.h"

struct LossSnap
{
    int epoch;
    double loss;
};
using PredictSnap = std::vector<double>;

using Losses = std::vector<LossSnap>;
using Predicts = std::vector<PredictSnap>;

class Trainer
{
public:
    Trainer(Network &net, double lr);

    void train(
        const std::vector<Sample> &data, int max_epochs,
        int sample_count, float x_min, float x_max);

private:
    Network &_net;
    double _lr;

public:
    Losses losses;
    Predicts predicts;
};

#endif //TRAINER_H