#include "MainWindow.h"

#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QSlider>
#include <QComboBox>
#include <QLabel>
#include <QFont>

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent),
      network(std::vector<int>{2, 4, 1}, /*seed=*/7) // 2 inputs -> 4 hidden -> 1 output
{
    // ---- XOR dataset ----
    trainingInputs  = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
    trainingTargets = { {0},    {1},    {1},    {0}    };

    // ---- Central widget & layout ----
    QWidget* central = new QWidget(this);
    QVBoxLayout* mainLayout = new QVBoxLayout(central);

    networkWidget = new NetworkWidget(central);
    networkWidget->setNetwork(&network);
    mainLayout->addWidget(networkWidget, /*stretch=*/1);

    // ---- Status label (epoch + loss) ----
    statusLabel = new QLabel(QStringLiteral("Epoch 0  |  Loss: --"), central);
    QFont statusFont = statusLabel->font();
    statusFont.setPointSize(11);
    statusLabel->setFont(statusFont);
    statusLabel->setAlignment(Qt::AlignCenter);
    mainLayout->addWidget(statusLabel);

    // ---- Controls row ----
    QHBoxLayout* controlsLayout = new QHBoxLayout();

    startPauseButton = new QPushButton(QStringLiteral("Start"), central);
    stepButton = new QPushButton(QStringLiteral("Step"), central);
    resetButton = new QPushButton(QStringLiteral("Reset"), central);

    controlsLayout->addWidget(startPauseButton);
    controlsLayout->addWidget(stepButton);
    controlsLayout->addWidget(resetButton);

    controlsLayout->addSpacing(20);

    learningRateLabel = new QLabel(QStringLiteral("Learning rate: 0.50"), central);
    learningRateSlider = new QSlider(Qt::Horizontal, central);
    learningRateSlider->setRange(1, 200);     // maps to 0.01 .. 2.00
    learningRateSlider->setValue(50);          // 0.50 default
    learningRateSlider->setFixedWidth(160);

    controlsLayout->addWidget(learningRateLabel);
    controlsLayout->addWidget(learningRateSlider);

    controlsLayout->addSpacing(20);

    edgeModeCombo = new QComboBox(central);
    edgeModeCombo->addItem(QStringLiteral("Edges: Weights"));
    edgeModeCombo->addItem(QStringLiteral("Edges: Gradients"));

    controlsLayout->addWidget(edgeModeCombo);
    controlsLayout->addStretch(1);

    mainLayout->addLayout(controlsLayout);

    setCentralWidget(central);
    resize(900, 650);
    setWindowTitle(QStringLiteral("Neural Network Visualizer"));

    // ---- Signal / slot wiring ----
    connect(startPauseButton, &QPushButton::clicked, this, &MainWindow::onStartPauseClicked);
    connect(stepButton, &QPushButton::clicked, this, &MainWindow::onStepClicked);
    connect(resetButton, &QPushButton::clicked, this, &MainWindow::onResetClicked);
    connect(learningRateSlider, &QSlider::valueChanged, this, &MainWindow::onLearningRateChanged);
    connect(edgeModeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MainWindow::onEdgeModeChanged);
    connect(&trainingTimer, &QTimer::timeout, this, &MainWindow::onTimerTick);

    trainingTimer.setInterval(400); // ms between training steps while running

    // Do one forward pass up front so the network isn't blank on launch.
    network.forward(trainingInputs[0]);
    networkWidget->update();
}

void MainWindow::onStartPauseClicked() {
    isRunning = !isRunning;
    if (isRunning) {
        startPauseButton->setText(QStringLiteral("Pause"));
        trainingTimer.start();
    } else {
        startPauseButton->setText(QStringLiteral("Start"));
        trainingTimer.stop();
    }
}

void MainWindow::onStepClicked() {
    // Manual single-step also pauses continuous running, so the
    // user can click Step repeatedly without fighting the timer.
    if (isRunning) {
        isRunning = false;
        startPauseButton->setText(QStringLiteral("Start"));
        trainingTimer.stop();
    }
    runOneTrainingStep();
}

void MainWindow::onResetClicked() {
    isRunning = false;
    startPauseButton->setText(QStringLiteral("Start"));
    trainingTimer.stop();

    network.reset(/*seed=*/static_cast<unsigned>(epoch + 7));
    epoch = 0;
    currentExampleIndex = 0;

    network.forward(trainingInputs[0]);
    networkWidget->setActiveLayer(-1);
    networkWidget->update();
    refreshStatusLabel(-1.0);
}

void MainWindow::onLearningRateChanged(int sliderValue) {
    learningRate = sliderValue / 100.0;
    learningRateLabel->setText(QStringLiteral("Learning rate: %1")
                                    .arg(learningRate, 0, 'f', 2));
}

void MainWindow::onEdgeModeChanged(int index) {
    networkWidget->setEdgeMode(index == 0 ? NetworkWidget::EdgeMode::Weight
                                           : NetworkWidget::EdgeMode::Gradient);
}

void MainWindow::onTimerTick() {
    runOneTrainingStep();
}

void MainWindow::runOneTrainingStep() {
    const std::vector<double>& inputs = trainingInputs[currentExampleIndex];
    const std::vector<double>& targets = trainingTargets[currentExampleIndex];

    network.forward(inputs);
    double loss = network.backward(targets, learningRate);

    currentExampleIndex = (currentExampleIndex + 1) % trainingInputs.size();
    if (currentExampleIndex == 0) {
        ++epoch;
    }

    networkWidget->update();
    refreshStatusLabel(loss);
}

void MainWindow::refreshStatusLabel(double loss) {
    if (loss < 0.0) {
        statusLabel->setText(QStringLiteral("Epoch 0  |  Loss: --"));
    } else {
        statusLabel->setText(QStringLiteral("Epoch %1  |  Loss: %2")
                                  .arg(epoch)
                                  .arg(loss, 0, 'f', 5));
    }
}
