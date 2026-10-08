#pragma once

#include <QWidget>
#include <QColor>
#include <vector>
#include "Network.h"

// ============================================================
//  NetworkWidget.h
//  A QWidget that paints the current state of a Network:
//    - neurons as circles, filled by activation value
//    - connections as lines, colored/thickened by weight
//      and (once backprop runs) by gradient magnitude
//
//  This widget does NOT own the Network's training logic.
//  It just reads whatever Network it's pointed at and paints
//  it. MainWindow is responsible for calling forward()/backward()
//  and then calling update() on this widget.
// ============================================================
class NetworkWidget : public QWidget {
    Q_OBJECT

public:
    explicit NetworkWidget(QWidget* parent = nullptr);

    // Point this widget at the network to visualize. Does not take ownership.
    void setNetwork(Network* net);

    // Controls what the edge color/thickness represents.
    enum class EdgeMode { Weight, Gradient };
    void setEdgeMode(EdgeMode mode);

    // Highlights the layer currently being processed (for step animation).
    // -1 means "no highlight".
    void setActiveLayer(int layerIndex);

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    Network* network = nullptr;
    EdgeMode edgeMode = EdgeMode::Weight;
    int activeLayer = -1;

    // Layout helpers: compute pixel position of neuron (layerIdx, neuronIdx)
    QPointF neuronPos(int layerIdx, int neuronIdx, int neuronsInLayer) const;

    // Map an activation value in [0,1] to a fill color (blue -> orange ramp).
    static QColor activationToColor(double activation);

    // Map a weight (-1..1 typical range, but unbounded) to an edge color.
    static QColor weightToColor(double weight);

    // Map a gradient magnitude to an edge color (used in backprop visualization).
    static QColor gradientToColor(double gradient);

    static constexpr int kNeuronRadius = 18;
    static constexpr int kMargin = 60;
};
