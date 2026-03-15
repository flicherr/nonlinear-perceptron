#include <random>
#include <omp.h>
#include "dense.h"

Dense::Dense(size_t in_size, size_t out_size) :
    _in_size(static_cast<int>(in_size)),
    _out_size(static_cast<int>(out_size))
{
    _w.resize(_out_size, std::vector<double>(_in_size));
    _b.resize(_out_size);

    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution dist(
        -sqrt(1.0/_in_size), sqrt(1.0/_in_size));

    for (int j = 0; j < _out_size; ++j) {
        for (int i = 0; i < _in_size; ++i) {
            _w[j][i] = dist(gen);
        }
        _b[j] = dist(gen);
    }
}

std::vector<double> Dense::forward(
    const std::vector<double>& input) // x
{
    _input = input;
    std::vector<double> output(_out_size);

    #pragma omp parallel for
    for (int j = 0; j < _out_size; ++j) {
        double sum = _b[j];

        for (int i = 0; i < _in_size; ++i) {
            sum += _w[j][i] * input[i];
        }

        output[j] = sum;
    }
    return output;
}

std::vector<double> Dense::backward(
    const std::vector<double> &grad_output, double lr)
{
    std::vector grad_input(_in_size, 0.0);

    #pragma omp parallel for
    for (int i = 0; i < _in_size; ++i) {
        double cur_grad = 0.0;

        for (int j = 0; j < _out_size; ++j) {
            cur_grad += grad_output[j] * _w[j][i];
        }

        grad_input[i] = cur_grad;
    }

    #pragma omp parallel for
    for (int j = 0; j < _out_size; ++j) {
        for (int i = 0; i < _in_size; ++i) {
            double grad_w = grad_output[j] * _input[i];
            _w[j][i] -= lr * grad_w;
        }

        _b[j] -= lr * grad_output[j];
    }

    return grad_input;
}