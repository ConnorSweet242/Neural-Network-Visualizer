cmake_minimum_required(VERSION 3.16)
project(NeuralNetworkVisualizer VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Qt-specific settings: automatically run moc/uic/rcc as needed.
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt6 REQUIRED COMPONENTS Widgets)

set(SOURCES
    main.cpp
    MainWindow.cpp
    NetworkWidget.cpp
)

set(HEADERS
    MainWindow.h
    NetworkWidget.h
    Network.h
)

add_executable(NeuralNetworkVisualizer ${SOURCES} ${HEADERS})

target_link_libraries(NeuralNetworkVisualizer PRIVATE Qt6::Widgets)
