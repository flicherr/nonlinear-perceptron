#include <iostream>
#include <cmath>

#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <glad/glad.h>

#include "gui_manager.h"

gui::Manager::~Manager()
{
    cleanup();
}

bool gui::Manager::init()
{
    if (!init_glfw()) {
        return false;
    }
    init_imgui();
    return true;
}

bool gui::Manager::init_glfw()
{
    glfwSetErrorCallback([](int error, const char* desc){
        std::cerr << "GLFW Error " << error << ": " << desc << "\n";
    });

    if (!glfwInit()) return false;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    _window = glfwCreateWindow(
        1000, 780, "nonlinear-perceptron",
        nullptr, nullptr);
    if (!_window) return false;
    glfwMakeContextCurrent(_window);
    glfwSwapInterval(1);

    if (!gladLoadGL()) {
        std::cerr << "Failed to initialize GLAD\n";
        return false;
    }

    return true;
}

void gui::Manager::init_imgui() const
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiStyle &style = ImGui::GetStyle();

    style.WindowRounding = 8.0f;
    style.FrameRounding  = 6.0f;
    style.ScrollbarRounding = 6.0f;
    style.FrameBorderSize  = 1.0f;
    style.WindowBorderSize = 1.0f;
    style.WindowPadding = ImVec2(12, 12);
    style.FramePadding  = ImVec2(8, 4);
    style.ItemSpacing   = ImVec2(8, 6);
    style.ScaleAllSizes(1.0f);

    ImGuiIO &io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(_window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
}


void gui::Manager::cleanup() const
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(_window);
    glfwTerminate();
}

void gui::Manager::show() {
	while (!glfwWindowShouldClose(_window)) {
		glfwPollEvents();

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		draw_toolbar();
		draw_canvas();

		ImGui::Render();
		int display_w, display_h;
		glfwGetFramebufferSize(
			_window, &display_w, &display_h);
		glViewport(0, 0, display_w, display_h);
		glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		ImGui_ImplOpenGL3_RenderDrawData(
			ImGui::GetDrawData());
		glfwSwapBuffers(_window);
	}
}

void gui::Manager::set_appstate(AppState &&app_state)
{
	_state = std::move(app_state);
}

ImVec2 gui::Manager::world2screen(
    double x, double y,ImVec2 origin, ImVec2 size) const
{
    return {
        origin.x + size.x * 0.5f +
            static_cast<float>(x * _state.scale) + _state.offset.x,
        origin.y + size.y * 0.5f -
            static_cast<float>(y * _state.scale) + _state.offset.y
    };
}

void gui::Manager::draw_canvas()
{
    ImGui::Begin("Visualization");

	ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
	ImVec2 canvas_size = ImGui::GetContentRegionAvail();
	if (canvas_size.x < 50) canvas_size.x = 50;
	if (canvas_size.y < 50) canvas_size.y = 50;

	ImDrawList *draw_list = ImGui::GetWindowDrawList();

	draw_list->AddRectFilled(
		canvas_pos,
		ImVec2(canvas_pos.x + canvas_size.x,
			canvas_pos.y + canvas_size.y),
		IM_COL32(40, 40, 40, 255)
	);

	draw_list->AddRect(
		canvas_pos,
		ImVec2(canvas_pos.x + canvas_size.x,
			canvas_pos.y + canvas_size.y),
		IM_COL32(255, 255, 255, 255)
	);

	for (float x = fmodf(canvas_pos.x, 30);
		x < canvas_size.x; x += 30) {
		draw_list->AddLine({canvas_pos.x + x, canvas_pos.y},
			{canvas_pos.x + x, canvas_pos.y + canvas_size.y - 2},
			IM_COL32(60,60,60,80));
	}

	for (float y = fmodf(canvas_pos.y, 30);
		y < canvas_size.y; y += 30) {
		draw_list->AddLine({canvas_pos.x, canvas_pos.y + y},
			{canvas_pos.x + canvas_size.x, canvas_pos.y + y - 2},
			IM_COL32(60,60,60,80));
	}

	ImGui::InvisibleButton("canvas", canvas_size);
	const ImGuiIO &io = ImGui::GetIO();
	if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
		const ImVec2 drag_delta = io.MouseDelta;
		_state.offset.x += drag_delta.x;
		_state.offset.y += drag_delta.y;
	}

	if (ImGui::IsItemHovered()) {
		constexpr float zoomSpeed = 0.1f;
		_state.scale *= 1.0f + io.MouseWheel * zoomSpeed;
	}

	// train points
	for (const auto & [x, y] : _state.data.train) {
		ImVec2 p = world2screen(x, y, canvas_pos, canvas_size);
		draw_list->AddCircleFilled(p, 3.5f, IM_COL32(80, 80, 255, 255));
	}
	// test points
	for (const auto & [x, y] : _state.data.test) {
		ImVec2 p = world2screen(x, y, canvas_pos, canvas_size);
		draw_list->AddCircleFilled(p, 3.5f, IM_COL32(200, 200, 80, 255));
	}

	// line
	ImVec2 prev;

	for(int i = 0; i < _state.sample_count; ++i) {
		double t = static_cast<double>(i) /
			(_state.sample_count - 1);

		const double x =
			_state.x_min + t*(_state.x_max-_state.x_min);
		const double y =
			_state.data.predicts[_state.cur_epoch][i];

		ImVec2 p = world2screen(
			x, y, canvas_pos, canvas_size);

		if(i > 0) {
			draw_list->AddLine(
				prev, p,
				IM_COL32(220,220,220,255),
				2.0f);
		}

		prev = p;
	}

	ImGui::End();
}

void gui::Manager::draw_toolbar()
{
	ImGui::Begin("Controls", nullptr,
		ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoScrollbar);

	ImGui::BeginGroup();
	if (_state.is_playing) {
		if (ImGui::Button("Pause"))
			_state.is_playing = false;
	} else {
		if (ImGui::Button("Play"))
			_state.is_playing = true;
	}
	ImGui::EndGroup();

	ImGui::SameLine(68.0f);

	ImGui::BeginGroup();
	if (ImGui::Button("Reset")) {
		_state.cur_epoch = 0;
		_state.is_playing = false;
	}
	ImGui::EndGroup();

	const int epochs =
		static_cast<int>(_state.data.predicts.size());

	constexpr float separator = 130.0f;
	ImGui::SameLine(separator);

	ImGui::BeginGroup();
	ImGui::SliderInt("Epochs", &_state.cur_epoch,
		0, epochs - 1);
	ImGui::EndGroup();

	ImGui::BeginGroup();
	ImGui::Text("loss = %.5f;",
		_state.data.losses[_state.cur_epoch]);
	ImGui::EndGroup();

	ImGui::SameLine(separator);

	ImGui::SliderInt("Speed (epoch/sec)",
		&_state.play_speed, 1, 40);

	ImGui::End();

	if (_state.is_playing) {
		if (const double currentTime = glfwGetTime();
			currentTime - _state.last_update_time >=
			1.0 / _state.play_speed) {
			++_state.cur_epoch;

			if (_state.cur_epoch >= epochs) {
				_state.cur_epoch = epochs - 1;
				_state.is_playing = false;
			}
			_state.last_update_time = currentTime;
		}
	}
}