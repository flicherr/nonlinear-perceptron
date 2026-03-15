#ifndef LAYER_H
#define LAYER_H

#include <vector>

class Layer
{
public:
    virtual ~Layer() = default;

    virtual std::vector<double> forward(
        const std::vector<double> &input) = 0;

    virtual std::vector<double> backward(
        const std::vector<double> &grad_output, double lr) = 0;
};

#endif //LAYER_H