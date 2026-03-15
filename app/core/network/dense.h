#ifndef DENSE_H
#define DENSE_H

#include "layer.h"

class Dense final : public Layer
{
public:
    Dense(size_t in_size, size_t out_size);

    std::vector<double> forward(
        const std::vector<double> &input) override;

    std::vector<double> backward(
        const std::vector<double> &grad_output, double lr) override;

private:
    int _in_size;
    int _out_size;

    std::vector<std::vector<double>> _w;
    std::vector<double> _b;

    std::vector<double> _input;
};

#endif //DENSE_H