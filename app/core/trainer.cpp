#include <omp.h>
#include "trainer.h"

Trainer::Trainer(Network &net, double lr)
    : _net(net), _lr(lr) {}

void Trainer::train(
    const std::vector<Sample> &data, int max_epochs,
    int sample_count, float x_min, float x_max)
{
    losses.clear();
    predicts.clear();

    predicts.reserve(max_epochs);
    losses.reserve(max_epochs);

    double prev_error = 1e18;
    constexpr double delta = 1e-10;
    constexpr double patience = 30;

    int bad_epochs = 0;

    for (int epoch = 0; epoch < max_epochs; ++epoch) {
        double total_error = 0;

        #pragma omp parallel for reduction(+:total_error)
        for (const auto &[x, y] : data) {
            std::vector input = {x};
            auto output = _net.forward(input);

            double y_pred = output[0];
            double y_true = y;

            double error = y_pred - y_true;
            total_error += error * error;

            std::vector grad = {2 * error};
            _net.backward(grad, _lr);
        }

        PredictSnap predict_snap;
        #pragma omp parallel for reduction(+:predict_snap)
        for (int i = 0; i < sample_count; ++i) {
            double t = static_cast<double>(i)/(sample_count - 1);
            double x = x_min + t*(x_max-x_min);

            std::vector input = {x};
            auto output = _net.forward(input);

            double y_pred = output[0];
            predict_snap.push_back(y_pred);
        }
        predicts.push_back(predict_snap);

        const double loss =
            total_error / static_cast<double>(data.size());
        losses.push_back({epoch, loss});

        if (const double diff = prev_error - loss; diff * diff < delta) {
            ++bad_epochs;
        } else {
            bad_epochs = 0;
        }

        if (bad_epochs >= patience) {
            break;
        }

        prev_error = loss;
    }
}