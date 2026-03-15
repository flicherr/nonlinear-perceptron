#include "network.h"

void Network::add_layer(std::unique_ptr<Layer> layer)
{
    _layers.push_back(std::move(layer));
}

std::vector<double> Network::forward(
    const std::vector<double> &input) const
{
    std::vector<double> x = input;

    for (auto &layer : _layers) {
        x = layer->forward(x);
    }
    return x;
}

void Network::backward(
    const std::vector<double> &grad, double lr) const
{
    std::vector<double> g = grad;
    for (int i = static_cast<int>(_layers.size()) - 1; i >= 0; --i) {
        g = _layers[i]->backward(g, lr);
    }
}