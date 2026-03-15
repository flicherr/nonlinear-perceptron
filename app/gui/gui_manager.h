#ifndef GUI_MANAGER_H
#define GUI_MANAGER_H

#include <GLFW/glfw3.h>
#include <imgui.h>
#include "appstate.h"

namespace gui
{
    class Manager
    {
    public:
        ~Manager();

        bool init();

        void set_appstate(AppState &&app_state);
        void show();
    private:
        bool init_glfw();
        void init_imgui() const;

        void cleanup() const;

        void draw_canvas();
        void draw_toolbar();

        ImVec2 world2screen(
            double x, double y,
            ImVec2 origin, ImVec2 size) const;

    private:
        GLFWwindow *_window = nullptr;
        AppState _state;
    };

}

#endif //GUI_MANAGER_H