#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>

#include <mujoco/mujoco.h>
#include <unitree/robot/channel/channel_factory.hpp>

#include "unitree_sdk2_bridge.h"

namespace {

bool finite_sensor_data(const mjModel* model, const mjData* data)
{
    for (int index = 0; index < model->nsensordata; ++index) {
        if (!std::isfinite(data->sensordata[index])) {
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "usage: go2_unitree_mujoco_headless_smoke SCENE_XML\n";
        return EXIT_FAILURE;
    }

    char error[1024] = {};
    std::unique_ptr<mjModel, decltype(&mj_deleteModel)> model(
        mj_loadXML(argv[1], nullptr, error, sizeof(error)), mj_deleteModel);
    if (!model) {
        std::cerr << "failed to load Go2 scene: " << error << '\n';
        return EXIT_FAILURE;
    }
    std::unique_ptr<mjData, decltype(&mj_deleteData)> data(
        mj_makeData(model.get()), mj_deleteData);
    if (!data || model->nu != 12) {
        std::cerr << "Go2 scene must expose exactly 12 actuators\n";
        return EXIT_FAILURE;
    }

    param::config.robot = "go2";
    param::config.robot_scene = argv[1];
    param::config.domain_id = 97;
    param::config.interface = "lo";
    param::config.use_joystick = 0;
    param::config.print_scene_information = 0;
    param::config.enable_elastic_band = 0;
    unitree::robot::ChannelFactory::Instance()->Init(
        param::config.domain_id, param::config.interface);

    mj_forward(model.get(), data.get());
    Go2Bridge bridge(model.get(), data.get());
    {
        std::lock_guard<std::mutex> lock(bridge.lowcmd->mutex_);
        for (int index = 0; index < model->nu; ++index) {
            auto& command = bridge.lowcmd->msg_.motor_cmd()[index];
            command.mode() = 1;
            command.q() = data->sensordata[index];
            command.dq() = 0.0F;
            command.kp() = index % 3 == 2 ? 40.0F : 20.0F;
            command.kd() = index % 3 == 2 ? 2.0F : 1.0F;
            command.tau() = 0.0F;
        }
        bridge.lowcmd->msg_.motor_cmd()[0].q() += 0.01F;
    }

    for (int step = 0; step < 10; ++step) {
        bridge.run();
        for (int index = 0; index < model->nu; ++index) {
            if (!std::isfinite(data->ctrl[index])) {
                std::cerr << "bridge generated non-finite actuator control\n";
                return EXIT_FAILURE;
            }
        }
        mj_step(model.get(), data.get());
        if (!finite_sensor_data(model.get(), data.get())) {
            std::cerr << "MuJoCo generated non-finite sensor data\n";
            return EXIT_FAILURE;
        }
    }

    bridge.run();
    const auto& lowstate = bridge.lowstate->msg_;
    if (lowstate.tick() == 0) {
        std::cerr << "bridge did not advance the SDK LowState tick\n";
        return EXIT_FAILURE;
    }
    for (int index = 0; index < model->nu; ++index) {
        const auto& motor = lowstate.motor_state()[index];
        if (!std::isfinite(motor.q()) || !std::isfinite(motor.dq()) ||
            !std::isfinite(motor.tau_est())) {
            std::cerr << "bridge generated non-finite SDK motor state\n";
            return EXIT_FAILURE;
        }
    }
    for (float value : lowstate.imu_state().quaternion()) {
        if (!std::isfinite(value)) {
            std::cerr << "bridge generated non-finite SDK IMU state\n";
            return EXIT_FAILURE;
        }
    }

    std::cout << "Go2 unitree_mujoco headless bridge smoke: PASS\n";
    return EXIT_SUCCESS;
}
