add_library(core
        network/dense.cpp
        network/tanh.cpp
        network/network.cpp
        trainer.cpp
        data/dataset.cpp
)
target_include_directories(core PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/data
)
target_link_libraries(core PUBLIC)