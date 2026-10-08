#pragma once

#include <QMainWindow>
#include <QTimer>
#include <QLabel>
#include <vector>
#include "Network.h"
#include "NetworkWidget.h"

class QPushButton;
class QSlider;
class QComboBox;

// ============================================================
//  MainWindow.h
//  Owns the Network and the training loop. Wires up:
//    - Start / Pause   (QTimer drives continuous training)
//    - Step            (single forward+backward pass)
//    - Reset           (re-randomize weights)
//    - Learning rate slider
//    - Edge display mode (weights vs gradients)
//
//  Training task: XOR. It's the classic "small but genuinely
//  not linearly separable" problem, so watching it learn is
//  actually convincing -- and it only needs a tiny network
//  (2 inputs, one hidden layer, 1 output), which keeps the
//  visualization readable.
// ============================================================
class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);

private slots:
    void onStartPauseClicked();
    void onStepClicked();
    void onResetClicked();
    void onLearningRateChanged(int sliderValue);
    void onEdgeModeChanged(int index);
    void onTimerTick();

private:
    void runOneTrainingStep();
    void refreshStatusLabel(double loss);

    Network network;
    NetworkWidget* networkWidget;

    QPushButton* startPauseButton;
    QPushButton* stepButton;
    QPushButton* resetButton;
    QSlider* learningRateSlider;
    QLabel* learningRateLabel;
    QComboBox* edgeModeCombo;
    QLabel* statusLabel;

    QTimer trainingTimer;
    bool isRunning = false;
    double learningRate = 0.5;
    int epoch = 0;

    // XOR training set: 4 examples, 2 inputs each, 1 target output.
    std::vector<std::vector<double>> trainingInputs;
    std::vector<std::vector<double>> trainingTargets;
    size_t currentExampleIndex = 0;
};
