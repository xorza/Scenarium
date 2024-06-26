#pragma once

#include "ArgumentController.hpp"

#include "../src/utils/uuid.hpp"

#include <QtCore>
#include <QQuickItem>


class NodeController : public QObject {
Q_OBJECT

    Q_PROPERTY(QString name READ name WRITE setName NOTIFY nameChanged)
    Q_PROPERTY(QList<ArgumentController *> inputs READ inputs NOTIFY inputsChanged)
    Q_PROPERTY(QList<ArgumentController *> outputs READ outputs NOTIFY outputsChanged)
    Q_PROPERTY(QList<ArgumentController *> events READ events NOTIFY eventsChanged)
    Q_PROPERTY(ArgumentController *trigger READ trigger NOTIFY triggerChanged)

    Q_PROPERTY(QPointF viewPos READ viewPos WRITE setViewPos NOTIFY viewPosChanged)
    Q_PROPERTY(QQuickItem *item READ item WRITE setItem NOTIFY itemChanged)

    Q_PROPERTY(bool selected READ selected WRITE setSelected NOTIFY selectedChanged)
    Q_PROPERTY(bool cacheResults READ cacheResults WRITE setCacheResults NOTIFY cacheResultsChanged)
    Q_PROPERTY(bool output READ output WRITE setOutput NOTIFY outputChanged)
public:
    explicit NodeController(QObject *parent = nullptr);

    ~NodeController() override = default;

    [[nodiscard]] QString name() const {
        return m_name;
    }

    void setName(const QString &name);

    [[nodiscard]] const QList<ArgumentController *> &inputs() const {
        return m_inputs;
    }

    void addInput(ArgumentController *input);

    [[nodiscard]] const QList<ArgumentController *> &outputs() const {
        return m_outputs;
    }

    void addOutput(ArgumentController *output);

    [[nodiscard]] QPointF viewPos() const {
        return m_viewPos;
    }

    void setViewPos(const QPointF &viewPos);

    [[nodiscard]] QQuickItem *item() const {
        return m_item;
    }

    void setItem(QQuickItem *item);


    [[nodiscard]] const QList<ArgumentController *> &events() const {
        return m_events;
    }

    void addEvent(ArgumentController *event);


    [[nodiscard]] bool selected() const {
        return m_selected;
    }

    void setSelected(bool selected);

    [[nodiscard]] ArgumentController *trigger() const {
        return m_trigger;
    }

    [[nodiscard]] uuid id() const {
        return m_id;
    }

    void updateViewPos(QPointF mousePos) const;

    [[nodiscard]] bool cacheResults() const {
        return m_cacheResults;
    }

    void setCacheResults(bool cacheResults);

    [[nodiscard]] bool output() const {
        return m_output;
    }

    void setOutput(bool output);

signals:

    void nameChanged();

    void inputsChanged();

    void outputsChanged();

    void viewPosChanged();

    void itemChanged();

    void eventsChanged();

    void selectedChanged();

    void triggerChanged();

    void selectedArgumentChanged(ArgumentController *arg);

    void removeRequested();

    void cacheResultsChanged();

    void outputChanged();

public slots:

    void remove();

private:
    QString m_name{};
    QList<ArgumentController *> m_inputs{};
    QList<ArgumentController *> m_outputs{};
    QPointF m_viewPos{};
    QQuickItem *m_item{};
    QList<ArgumentController *> m_events{};
    bool m_selected = false;
    ArgumentController *m_trigger{};
    uuid m_id{};
    bool m_cacheResults = false;
    bool m_output = false;
};
