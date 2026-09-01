#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());
    auto deploy_cfg = YAML::LoadFile(policy_dir / "params" / "deploy.yaml");

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        deploy_cfg,
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(
        policy_dir / "exported" / "policy.onnx",
        deploy_cfg["schema"]["sha256"].as<std::string>(),
        deploy_cfg["schema"]["actor_dim"].as<size_t>(),
        deploy_cfg["schema"]["action_dim"].as<size_t>(),
        deploy_cfg["schema"]["action_interface"]
            ? deploy_cfg["schema"]["action_interface"].as<std::string>() : "",
        deploy_cfg["schema"]["action_output_semantics"]
            ? deploy_cfg["schema"]["action_output_semantics"].as<std::string>() : "");

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return env->runtime_fault.load(); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    if (env->runtime_fault.load() || !env->action_ready.load()) {
        return;
    }
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}
