// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <eigen3/Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include "isaaclab/manager/observation_manager.h"
#include "isaaclab/manager/action_manager.h"
#include "isaaclab/assets/articulation/articulation.h"
#include "isaaclab/algorithms/algorithms.h"
#include <iostream>
#include "isaaclab/utils/utils.h"
#include <atomic>
#include <cmath>

namespace isaaclab
{

class ObservationManager;
class ActionManager;

class ManagerBasedRLEnv
{
public:
    // Constructor
    ManagerBasedRLEnv(YAML::Node cfg, std::shared_ptr<Articulation> robot_)
    :cfg(cfg), robot(std::move(robot_))
    {
        // Parse configuration
        this->step_dt = cfg["step_dt"].as<float>();
        robot->data.joint_ids_map = cfg["joint_ids_map"].as<std::vector<float>>();
        robot->data.joint_pos.resize(robot->data.joint_ids_map.size());
        robot->data.joint_vel.resize(robot->data.joint_ids_map.size());

        { // default joint positions
            auto default_joint_pos = cfg["default_joint_pos"].as<std::vector<float>>();
            robot->data.default_joint_pos = Eigen::VectorXf::Map(default_joint_pos.data(), default_joint_pos.size());
        }
        { // joint stiffness and damping
            robot->data.joint_stiffness = cfg["stiffness"].as<std::vector<float>>();
            robot->data.joint_damping = cfg["damping"].as<std::vector<float>>();
        }

        robot->update();

        // load managers
        action_manager = std::make_unique<ActionManager>(cfg["actions"], this);
        observation_manager = std::make_unique<ObservationManager>(cfg["observations"], this);
    }

    void reset()
    {
        policy_tick = 0;
        episode_length = 0;
        runtime_fault.store(false);
        action_ready.store(false);
        robot->update();
        action_manager->reset();
        observation_manager->reset();
    }

    void step()
    {
        episode_length += 1;
        policy_tick += 1;
        robot->update();
        if (!robot_state_is_finite()) {
            runtime_fault.store(true);
            return;
        }
        auto obs = observation_manager->compute();
        for (const auto& group : obs) {
            for (float value : group.second) {
                if (!std::isfinite(value)) {
                    runtime_fault.store(true);
                    return;
                }
            }
        }
        std::vector<float> action;
        try {
            action = alg->act(obs);
        } catch (const std::exception&) {
            runtime_fault.store(true);
            return;
        }
        const float action_limit = cfg["safety"]["action_abs_limit"].as<float>();
        if (action.size() != static_cast<size_t>(action_manager->total_action_dim())) {
            runtime_fault.store(true);
            return;
        }
        for (float value : action) {
            if (!std::isfinite(value) || std::abs(value) > action_limit) {
                runtime_fault.store(true);
                return;
            }
        }
        action_manager->process_action(action);
        const auto processed = action_manager->processed_actions();
        const auto joint_limits = cfg["safety"]["joint_pos_limits"].as<
            std::vector<std::vector<float>>>();
        if (processed.size() != joint_limits.size()) {
            runtime_fault.store(true);
            return;
        }
        for (size_t index = 0; index < processed.size(); ++index) {
            if (joint_limits[index].size() != 2 ||
                !std::isfinite(processed[index]) ||
                processed[index] < joint_limits[index][0] ||
                processed[index] > joint_limits[index][1]) {
                runtime_fault.store(true);
                return;
            }
        }
        action_ready.store(true);
    }

    bool robot_state_is_finite() const
    {
        const auto& data = robot->data;
        if (!data.root_ang_vel_b.allFinite() || !data.projected_gravity_b.allFinite()) {
            return false;
        }
        if (!data.joint_pos.allFinite() || !data.joint_vel.allFinite()) {
            return false;
        }
        const float quat_norm = data.root_quat_w.norm();
        return std::isfinite(quat_norm) && quat_norm > 0.5f && quat_norm < 1.5f;
    }

    float step_dt;
    
    YAML::Node cfg;

    std::unique_ptr<ObservationManager> observation_manager;
    std::unique_ptr<ActionManager> action_manager;
    std::shared_ptr<Articulation> robot;
    std::unique_ptr<Algorithms> alg;
    long episode_length = 0;
    long policy_tick = 0;
    std::atomic<bool> runtime_fault{false};
    std::atomic<bool> action_ready{false};
};

};
