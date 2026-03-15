#include <iostream>
#include "dataset.h"
#include "gui_manager.h"
#include "network/dense.h"
#include "network/tanh.h"

void apply()
{
    AppState state;
    try {
        const std::vector<Sample> data = dataset::load_from_file(
            "scripts/dataset.csv");
        auto [train, test] = dataset::split(data);

        state.data.train = std::move(train);
        state.data.test = std::move(test);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
    }

    Network net;
    net.add_layer(
        std::make_unique<Dense>(1, state.neurons_count));
    net.add_layer(
        std::make_unique<Tanh>());
    net.add_layer(
        std::make_unique<Dense>(state.neurons_count, 1));

    Trainer trainer(net, state.lr);
    trainer.train(state.data.train, state.max_epochs,
        state.sample_count, state.x_min, state.x_max);

    state.data.losses = std::move(trainer.losses);
    state.data.predicts = std::move(trainer.predicts);

    gui::Manager mgr;
    mgr.init();
    mgr.set_appstate(std::move(state));

    mgr.show();
}

int main()
{
    apply();
    return 0;
}