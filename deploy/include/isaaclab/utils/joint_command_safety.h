// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#include "isaaclab/assets/articulation/articulation.h"

namespace isaaclab
{

template <typename MotorCommands>
bool initialize_measured_position_hold(
    const ArticulationData& data, MotorCommands& motor_commands)
{
    const std::size_t joint_count = data.joint_ids_map.size();
    const std::size_t motor_count = motor_commands.size();
    if (data.joint_stiffness.size() != joint_count ||
        data.joint_damping.size() != joint_count ||
        static_cast<std::size_t>(data.joint_pos.size()) != joint_count) {
        return false;
    }

    std::vector<bool> mapped_motors(motor_count, false);
    for (std::size_t index = 0; index < joint_count; ++index) {
        const float raw_sdk_id = data.joint_ids_map[index];
        const int sdk_id = static_cast<int>(raw_sdk_id);
        if (!std::isfinite(raw_sdk_id) || raw_sdk_id != sdk_id ||
            sdk_id < 0 || static_cast<std::size_t>(sdk_id) >= motor_count ||
            mapped_motors[sdk_id] || !std::isfinite(data.joint_pos[index]) ||
            !std::isfinite(data.joint_stiffness[index]) ||
            !std::isfinite(data.joint_damping[index])) {
            return false;
        }
        mapped_motors[sdk_id] = true;
    }

    for (std::size_t index = 0; index < joint_count; ++index) {
        const int sdk_id = static_cast<int>(data.joint_ids_map[index]);
        auto& motor = motor_commands[sdk_id];
        motor.q() = data.joint_pos[index];
        motor.kp() = data.joint_stiffness[index];
        motor.kd() = data.joint_damping[index];
        motor.dq() = 0.0f;
        motor.tau() = 0.0f;
    }
    return true;
}

}  // namespace isaaclab
