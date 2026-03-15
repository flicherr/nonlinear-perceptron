#include <cmath>
#include <omp.h>
#include "tanh.h"

std::vector<double> Tanh::forward(
    const std::vector<double> &input)
{
    _output.resize(input.size());

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(input.size()); ++i) {
        _output[i] = std::tanh(input[i]);
    }

    return _output;
}

std::vector<double> Tanh::backward(
    const std::vector<double> &grad_output, double lr)
{
    std::vector<double> grad_input(_output.size());

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(_output.size()); ++i) {
        double d = 1 - _output[i] * _output[i];
        grad_input[i] = grad_output[i] * d;
    }

    return grad_input;
}