#ifndef NETWORK_H
#define NETWORK_H

#include <memory>
#include <vector>
#include "layer.h"

class Network
{
public:
    void add_layer(std::unique_ptr<Layer> layer);

    std::vector<double> forward(
        const std::vector<double> &input) const;

    void backward(
        const std::vector<double> &grad, double lr) const;

private:
    std::vector<std::unique_ptr<Layer>> _layers;
};

#endif //NETWORK_H