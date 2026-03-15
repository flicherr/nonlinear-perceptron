#ifndef TANH_H
#define TANH_H

#include "layer.h"

class Tanh final : public Layer
{
public:
    std::vector<double> forward(
        const std::vector<double> &input) override;

    std::vector<double> backward(
        const std::vector<double> &grad_output, double lr) override;

private:
    std::vector<double> _output;
};

#endif //TANH_H