#include <yaml-cpp/yaml.h>

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/utils/joint_command_safety.h"

namespace
{

constexpr std::size_t kActorDim = 425;
constexpr std::size_t kActionDim = 12;

class MockArticulation final : public isaaclab::Articulation
{
public:
    MockArticulation()
    {
        data.joystick = &joystick;
        set_finite_state();
    }

    void set_finite_state()
    {
        data.root_ang_vel_b = Eigen::Vector3f::Zero();
        data.projected_gravity_b = Eigen::Vector3f(0.0f, 0.0f, -1.0f);
        data.root_quat_w = Eigen::Quaternionf::Identity();
        if (data.joint_pos.size() > 0) {
            data.joint_pos.setZero();
            data.joint_vel.setZero();
        }
    }

    unitree::common::UnitreeJoystick joystick;
};

class MockAlgorithm final : public isaaclab::Algorithms
{
public:
    std::vector<float> act(
        std::unordered_map<std::string, std::vector<float>> obs) override
    {
        if (throw_on_act) {
            throw std::runtime_error("injected inference failure");
        }
        const auto actor = obs.find("actor");
        if (actor == obs.end()) {
            throw std::runtime_error("actor observation missing");
        }
        observations.push_back(actor->second);
        return next_action;
    }

    std::vector<float> next_action = std::vector<float>(kActionDim, 0.0f);
    std::vector<std::vector<float>> observations;
    bool throw_on_act = false;
};

struct MockMotorCommand
{
    float& q() { return q_value; }
    float& kp() { return kp_value; }
    float& kd() { return kd_value; }
    float& dq() { return dq_value; }
    float& tau() { return tau_value; }

    float q_value = -99.0f;
    float kp_value = -99.0f;
    float kd_value = -99.0f;
    float dq_value = -99.0f;
    float tau_value = -99.0f;
};

struct Fixture
{
    explicit Fixture(const std::string& deploy_yaml)
      : robot(std::make_shared<MockArticulation>()),
        env(YAML::LoadFile(deploy_yaml), robot),
        algorithm(new MockAlgorithm())
    {
        env.alg.reset(algorithm);
    }

    void reset_finite()
    {
        robot->set_finite_state();
        algorithm->throw_on_act = false;
        algorithm->next_action.assign(kActionDim, 0.0f);
        algorithm->observations.clear();
        env.reset();
    }

    std::shared_ptr<MockArticulation> robot;
    isaaclab::ManagerBasedRLEnv env;
    MockAlgorithm* algorithm;
};

int require(bool condition, int code)
{
    return condition ? 0 : code;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc != 2) return 2;
    Fixture fixture(argv[1]);

    isaaclab::ArticulationData hold_data;
    hold_data.joint_ids_map = {2.0f, 0.0f, 1.0f};
    hold_data.joint_pos = Eigen::Vector3f(0.2f, 0.0f, 0.1f);
    hold_data.joint_stiffness = {22.0f, 20.0f, 21.0f};
    hold_data.joint_damping = {2.2f, 2.0f, 2.1f};
    std::vector<MockMotorCommand> hold_commands(3);
    if (int code = require(
            isaaclab::initialize_measured_position_hold(hold_data, hold_commands), 3)) return code;
    if (int code = require(
            hold_commands[0].q_value == 0.0f && hold_commands[0].kp_value == 20.0f &&
            hold_commands[0].kd_value == 2.0f, 4)) return code;
    if (int code = require(
            hold_commands[1].q_value == 0.1f && hold_commands[1].kp_value == 21.0f &&
            hold_commands[1].kd_value == 2.1f, 5)) return code;
    if (int code = require(
            hold_commands[2].q_value == 0.2f && hold_commands[2].kp_value == 22.0f &&
            hold_commands[2].kd_value == 2.2f, 6)) return code;
    for (const auto& motor : hold_commands) {
        if (int code = require(motor.dq_value == 0.0f && motor.tau_value == 0.0f, 7)) return code;
    }
    hold_data.joint_ids_map[2] = 0.0f;
    if (int code = require(
            !isaaclab::initialize_measured_position_hold(hold_data, hold_commands), 8)) return code;

    if (int code = require(std::abs(fixture.env.step_dt - 0.02f) < 1.0e-7f, 10)) return code;
    if (int code = require(fixture.env.action_manager->total_action_dim() == 12, 11)) return code;

    fixture.reset_finite();
    if (int code = require(!fixture.env.action_ready.load(), 12)) return code;
    if (int code = require(!fixture.env.runtime_fault.load(), 13)) return code;

    fixture.robot->joystick.ly.smooth = 1.0f;
    fixture.robot->joystick.ly(0.5f);
    fixture.robot->data.root_ang_vel_b = Eigen::Vector3f(1.0f, 2.0f, 3.0f);
    fixture.algorithm->next_action.assign(kActionDim, 0.5f);
    fixture.env.step();
    if (int code = require(fixture.env.action_ready.load(), 14)) return code;
    if (int code = require(!fixture.env.runtime_fault.load(), 15)) return code;
    if (int code = require(fixture.algorithm->observations.size() == 1, 16)) return code;

    const auto& first_obs = fixture.algorithm->observations.front();
    if (int code = require(first_obs.size() == kActorDim, 17)) return code;
    // Reset backfills nine retained old frames; the first step appends the new frame.
    if (int code = require(first_obs[0] == 0.0f && first_obs[27] == 1.0f, 18)) return code;
    if (int code = require(first_obs[28] == 2.0f && first_obs[29] == 3.0f, 19)) return code;
    if (int code = require(first_obs[60] == 0.5f, 29)) return code;
    // The first policy observation must contain only reset-zero action history.
    for (std::size_t index = 305; index < kActorDim; ++index) {
        if (int code = require(first_obs[index] == 0.0f, 20)) return code;
    }
    const float expected_phase = 2.0f * static_cast<float>(M_PI) * (0.02f / 0.6f);
    if (int code = require(std::abs(first_obs[63] - std::sin(expected_phase)) < 1.0e-6f, 21)) return code;
    if (int code = require(std::abs(first_obs[64] - std::cos(expected_phase)) < 1.0e-6f, 22)) return code;

    fixture.env.step();
    const auto& second_obs = fixture.algorithm->observations.back();
    if (int code = require(second_obs.size() == kActorDim, 23)) return code;
    for (std::size_t index = 305; index < 413; ++index) {
        if (int code = require(second_obs[index] == 0.0f, 24)) return code;
    }
    for (std::size_t index = 413; index < kActorDim; ++index) {
        if (int code = require(second_obs[index] == 0.5f, 25)) return code;
    }
    const auto targets = fixture.env.action_manager->processed_actions();
    if (int code = require(targets.size() == kActionDim, 26)) return code;
    if (int code = require(std::abs(targets[0] - 0.025f) < 1.0e-7f, 27)) return code;
    if (int code = require(std::abs(targets[1] - 1.025f) < 1.0e-7f, 28)) return code;

    fixture.reset_finite();
    fixture.algorithm->next_action.resize(kActionDim - 1);
    fixture.env.step();
    if (int code = require(fixture.env.runtime_fault.load() && !fixture.env.action_ready.load(), 30)) return code;

    fixture.reset_finite();
    fixture.algorithm->next_action[0] = std::numeric_limits<float>::quiet_NaN();
    fixture.env.step();
    if (int code = require(fixture.env.runtime_fault.load() && !fixture.env.action_ready.load(), 31)) return code;

    fixture.reset_finite();
    fixture.algorithm->next_action[0] = 4.01f;
    fixture.env.step();
    if (int code = require(fixture.env.runtime_fault.load() && !fixture.env.action_ready.load(), 32)) return code;

    fixture.reset_finite();
    fixture.algorithm->next_action[0] = -4.0f;
    fixture.env.step();
    if (int code = require(fixture.env.runtime_fault.load() && !fixture.env.action_ready.load(), 33)) return code;

    fixture.reset_finite();
    fixture.algorithm->throw_on_act = true;
    fixture.env.step();
    if (int code = require(fixture.env.runtime_fault.load() && !fixture.env.action_ready.load(), 34)) return code;

    fixture.reset_finite();
    fixture.robot->data.joint_vel[0] = std::numeric_limits<float>::infinity();
    fixture.env.step();
    if (int code = require(fixture.env.runtime_fault.load() && !fixture.env.action_ready.load(), 35)) return code;

    fixture.reset_finite();
    fixture.robot->data.root_quat_w.coeffs().setZero();
    fixture.env.step();
    if (int code = require(fixture.env.runtime_fault.load() && !fixture.env.action_ready.load(), 36)) return code;

    return 0;
}
