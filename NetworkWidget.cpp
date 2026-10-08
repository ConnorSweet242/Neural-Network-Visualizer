#include "NetworkWidget.h"
#include <QPainter>
#include <QPen>
#include <QBrush>
#include <algorithm>
#include <cmath>

NetworkWidget::NetworkWidget(QWidget* parent) : QWidget(parent) {
    setMinimumSize(600, 400);
}

void NetworkWidget::setNetwork(Network* net) {
    network = net;
    update();
}

void NetworkWidget::setEdgeMode(EdgeMode mode) {
    edgeMode = mode;
    update();
}

void NetworkWidget::setActiveLayer(int layerIndex) {
    activeLayer = layerIndex;
    update();
}

QPointF NetworkWidget::neuronPos(int layerIdx, int neuronIdx, int neuronsInLayer) const {
    int numLayers = network->numLayers();

    double usableWidth = width() - 2.0 * kMargin;
    double usableHeight = height() - 2.0 * kMargin;

    double x = kMargin;
    if (numLayers > 1) {
        x = kMargin + usableWidth * (static_cast<double>(layerIdx) / (numLayers - 1));
    }

    // Center neurons vertically within the layer's column.
    double slot = usableHeight / std::max(1, neuronsInLayer);
    double y = kMargin + slot * (neuronIdx + 0.5);

    return QPointF(x, y);
}

QColor NetworkWidget::activationToColor(double activation) {
    // Clamp to [0, 1] just in case.
    double a = std::clamp(activation, 0.0, 1.0);

    // Ramp from cool blue (low activation) to warm orange (high activation).
    QColor low(52, 88, 196);    // blue
    QColor high(235, 140, 52);  // orange

    int r = static_cast<int>(low.red()   + (high.red()   - low.red())   * a);
    int g = static_cast<int>(low.green() + (high.green() - low.green()) * a);
    int b = static_cast<int>(low.blue()  + (high.blue()  - low.blue())  * a);
    return QColor(r, g, b);
}

QColor NetworkWidget::weightToColor(double weight) {
    // Positive weights -> greenish, negative weights -> reddish.
    // Magnitude controls saturation/opacity-like darkness.
    double magnitude = std::clamp(std::fabs(weight) / 2.0, 0.0, 1.0); // weights typically in [-2,2]-ish
    if (weight >= 0.0) {
        int g = static_cast<int>(120 + 100 * magnitude);
        return QColor(80, std::min(g, 255), 90);
    } else {
        int r = static_cast<int>(120 + 100 * magnitude);
        return QColor(std::min(r, 255), 80, 90);
    }
}

QColor NetworkWidget::gradientToColor(double gradient) {
    // Gradients can be small; scale up so the color ramp is visible,
    // then map magnitude to a purple->yellow "heat" scale.
    double magnitude = std::clamp(std::fabs(gradient) * 8.0, 0.0, 1.0);
    QColor low(90, 70, 140);     // muted purple = small gradient
    QColor high(240, 220, 40);   // yellow = large gradient
    int r = static_cast<int>(low.red()   + (high.red()   - low.red())   * magnitude);
    int g = static_cast<int>(low.green() + (high.green() - low.green()) * magnitude);
    int b = static_cast<int>(low.blue()  + (high.blue()  - low.blue())  * magnitude);
    return QColor(r, g, b);
}

void NetworkWidget::paintEvent(QPaintEvent* /*event*/) {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);

    painter.fillRect(rect(), QColor(24, 26, 32)); // dark background

    if (network == nullptr) {
        painter.setPen(QColor(200, 200, 200));
        painter.drawText(rect(), Qt::AlignCenter, QStringLiteral("No network loaded"));
        return;
    }

    int numLayers = network->numLayers();

    // ---- Draw connections first, so neurons render on top ----
    for (int l = 1; l < numLayers; ++l) {
        Layer& currLayer = network->layer(l);
        Layer& prevLayer = network->layer(l - 1);

        for (int j = 0; j < currLayer.size(); ++j) {
            Neuron& n = currLayer.neurons[static_cast<size_t>(j)];
            QPointF dst = neuronPos(l, j, currLayer.size());

            for (int i = 0; i < prevLayer.size(); ++i) {
                QPointF src = neuronPos(l - 1, i, prevLayer.size());

                double weight = n.incomingWeights[static_cast<size_t>(i)];
                double grad = n.weightGradients.empty() ? 0.0 : n.weightGradients[static_cast<size_t>(i)];

                QColor edgeColor = (edgeMode == EdgeMode::Weight)
                                        ? weightToColor(weight)
                                        : gradientToColor(grad);

                double thickness = (edgeMode == EdgeMode::Weight)
                                        ? std::clamp(std::fabs(weight) * 1.8, 0.6, 5.0)
                                        : std::clamp(std::fabs(grad) * 14.0, 0.6, 5.0);

                QPen pen(edgeColor);
                pen.setWidthF(thickness);
                painter.setPen(pen);
                painter.drawLine(src, dst);
            }
        }
    }

    // ---- Draw neurons on top of connections ----
    for (int l = 0; l < numLayers; ++l) {
        Layer& layerRef = network->layer(l);
        bool isActive = (l == activeLayer);

        for (int i = 0; i < layerRef.size(); ++i) {
            Neuron& n = layerRef.neurons[static_cast<size_t>(i)];
            QPointF pos = neuronPos(l, i, layerRef.size());

            QColor fill = activationToColor(n.activation);
            QPen outline(isActive ? QColor(255, 255, 255) : QColor(10, 10, 10));
            outline.setWidthF(isActive ? 3.0 : 1.5);

            painter.setPen(outline);
            painter.setBrush(QBrush(fill));
            painter.drawEllipse(pos, kNeuronRadius, kNeuronRadius);

            // Draw the activation value as text inside/below the neuron.
            painter.setPen(QColor(235, 235, 235));
            QString label = QString::number(n.activation, 'f', 2);
            painter.drawText(QRectF(pos.x() - 24, pos.y() - 8, 48, 16),
                              Qt::AlignCenter, label);
        }

        // Layer caption below the column.
        painter.setPen(QColor(160, 160, 170));
        QString caption = (l == 0) ? QStringLiteral("Input")
                          : (l == numLayers - 1) ? QStringLiteral("Output")
                          : QStringLiteral("Hidden %1").arg(l);
        QPointF topPos = neuronPos(l, 0, layerRef.size());
        painter.drawText(QRectF(topPos.x() - 50, kMargin - 40, 100, 20),
                          Qt::AlignCenter, caption);
    }
}
