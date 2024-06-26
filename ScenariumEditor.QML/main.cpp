#include "src/CoreContext.hpp"
#include "src/utils/uuid.hpp"

#include "qml/NodeController.hpp"
#include "qml/ArgumentController.hpp"
#include "qml/AppController.hpp"
#include "qml/ConnectionController.hpp"


#include <QGuiApplication>
#include <QQmlApplicationEngine>

#include <iostream>


int main(int argc, char *argv[]) {
    QGuiApplication app(argc, argv);
    QQmlApplicationEngine engine;
    QObject::connect(
            &engine,
            &QQmlApplicationEngine::objectCreationFailed,
            &app,
            []() {
                qDebug() << "QQmlApplicationEngine::objectCreationFailed";
                QCoreApplication::exit(-1);
            },
            Qt::QueuedConnection);

    auto *const appController = new AppController(&app);
    appController->loadSample();
    qmlRegisterType<NodeController>("com.cssodessa.NodeController", 1, 0, "NodeController");
    qmlRegisterType<ArgumentController>("com.cssodessa.ArgumentController", 1, 0, "ArgumentController");
    qmlRegisterType<ConnectionController>("com.cssodessa.ConnectionController", 1, 0, "ConnectionController");
    qmlRegisterSingletonInstance("com.cssodessa.AppController", 1, 0, "AppController", appController);

    engine.loadFromModule("scenarium_editor", "Main");

    int res = QGuiApplication::exec();
    return res;
}
