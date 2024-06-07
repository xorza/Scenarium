#pragma once

#include <QtCore>

class ArgumentController : public QObject {
Q_OBJECT

    Q_PROPERTY(QString name READ name NOTIFY nameChanged)

public:
    explicit ArgumentController(QObject *parent = nullptr) : QObject(parent) {}

    ~ArgumentController() override = default;

    [[nodiscard]] QString name() const {
        return m_name;
    }

    void setName(const QString &name);

signals:

    void nameChanged();

private:
    QString m_name;

};



class NodeController : public QObject {
Q_OBJECT

    Q_PROPERTY(QString name READ name WRITE setName NOTIFY nameChanged)
    Q_PROPERTY(QList<ArgumentController *> inputs READ inputs NOTIFY inputsChanged)
    Q_PROPERTY(QList<ArgumentController *> outputs READ outputs NOTIFY outputsChanged)

public:
    explicit NodeController(QObject *parent = nullptr) : QObject(parent) {}

    ~NodeController() override = default;

    [[nodiscard]] QString name() const {
        return m_name;
    }

    void setName(const QString &name);

    [[nodiscard]] QList<ArgumentController *> inputs() const {
        return m_inputs;
    }

    void addInput(ArgumentController *const input) {
        m_inputs.push_back(input);
        emit inputsChanged();
    }

    [[nodiscard]] QList<ArgumentController *> outputs() const {
        return m_outputs;
    }

    void addOutput(ArgumentController *const output) {
        m_outputs.push_back(output);
        emit outputsChanged();
    }

signals:

    void nameChanged();

    void inputsChanged();
    void outputsChanged();

private:
    QString m_name{};
    QList<ArgumentController *> m_inputs{};
    QList<ArgumentController *> m_outputs{};
};
