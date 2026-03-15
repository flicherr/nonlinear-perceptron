#ifndef APPSTATE_H
#define APPSTATE_H

#include <vector>
#include <imgui.h>
#include "trainer.h"
#include "data/sample.h"

struct VisualData
{
    std::vector<Sample> train;
    std::vector<Sample> test;
    Losses losses;
    Predicts predicts;
};

struct AppState
{
    /*
     * x_min, x_max - the range of the line construction based on the
     * predicted y_pred values obtained during the training process;
     * sample_count - the number of values in the range [x_min; x_max].
     */
    float x_min = -6., x_max = 6.;
    int sample_count = 200;

    VisualData data;
    int neurons_count = 5;
    int max_epochs = 6000;
    float lr = 0.008;

    int cur_epoch = 0;

    bool is_playing = false;
    int play_speed = 30;
    double last_update_time = 0.;

    ImVec2 offset = ImVec2(0.f, 0.f);
    float scale = 68.;
    bool is_dragging = false;
    ImVec2 drag_start;
};

#endif //APPSTATE_H