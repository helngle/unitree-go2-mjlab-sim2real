// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "onnxruntime_cxx_api.h"
#include <iostream>
#include <cmath>
#include <mutex>
#include <stdexcept>

namespace isaaclab
{

class Algorithms
{
public:
    virtual std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) = 0;

    std::vector<float> get_action()
    {
        std::lock_guard<std::mutex> lock(act_mtx_);
        return action;
    }
    
    std::vector<float> action;
protected:
    std::mutex act_mtx_;
};

class OrtRunner : public Algorithms
{
public:
    OrtRunner(
        std::string model_path,
        const std::string& expected_schema_hash = "",
        size_t expected_input_size = 0,
        size_t expected_output_size = 0,
        const std::string& expected_action_interface = "",
        const std::string& expected_action_output_semantics = "")
    {
        // Init Model
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnx_model");
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

        if (!expected_schema_hash.empty()) {
            auto metadata = session->GetModelMetadata();
            auto value = metadata.LookupCustomMetadataMapAllocated(
                "observation_schema_sha256", allocator);
            if (!value || expected_schema_hash != value.get()) {
                throw std::runtime_error("ONNX observation schema SHA256 mismatch");
            }
            if (!expected_action_interface.empty()) {
                auto interface_value = metadata.LookupCustomMetadataMapAllocated(
                    "action_interface", allocator);
                if (!interface_value || expected_action_interface != interface_value.get()) {
                    throw std::runtime_error("ONNX action interface mismatch");
                }
            }
            if (!expected_action_output_semantics.empty()) {
                auto semantics_value = metadata.LookupCustomMetadataMapAllocated(
                    "action_output_semantics", allocator);
                if (!semantics_value ||
                    expected_action_output_semantics != semantics_value.get()) {
                    throw std::runtime_error("ONNX action output semantics mismatch");
                }
            }
        }

        for (size_t i = 0; i < session->GetInputCount(); ++i) {
            Ort::TypeInfo input_type = session->GetInputTypeInfo(i);
            input_shapes.push_back(input_type.GetTensorTypeAndShapeInfo().GetShape());
            auto input_name = session->GetInputNameAllocated(i, allocator);
            input_names.push_back(input_name.release());
        }

        for (const auto& shape : input_shapes) {
            size_t size = 1;
            for (const auto& dim : shape) {
                if (dim <= 0) {
                    throw std::runtime_error("ONNX input shapes must be static and positive");
                }
                size *= dim;
            }
            input_sizes.push_back(size);
        }
        if (input_sizes.size() != 1 ||
            (expected_input_size > 0 && input_sizes[0] != expected_input_size)) {
            throw std::runtime_error("ONNX actor input dimension mismatch");
        }
        if (expected_input_size > 0 &&
            (input_shapes[0].size() != 2 || input_shapes[0][0] != 1 ||
             input_shapes[0][1] != static_cast<int64_t>(expected_input_size))) {
            throw std::runtime_error("ONNX actor input shape must be static [1,N]");
        }

        // Get output shape
        if (session->GetOutputCount() != 1) {
            throw std::runtime_error("ONNX policy must have exactly one output");
        }
        Ort::TypeInfo output_type = session->GetOutputTypeInfo(0);
        output_shape = output_type.GetTensorTypeAndShapeInfo().GetShape();
        if (output_shape.size() != 2 || output_shape[0] != 1 || output_shape[1] <= 0) {
            throw std::runtime_error("ONNX action output shape must be static [1,N]");
        }
        if (expected_output_size > 0 &&
            static_cast<size_t>(output_shape[1]) != expected_output_size) {
            throw std::runtime_error("ONNX action output dimension mismatch");
        }
        auto output_name = session->GetOutputNameAllocated(0, allocator);
        output_names.push_back(output_name.release());

        action.resize(output_shape[1]);
    }

    std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs)
    {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        // make sure all input names are in obs
        for (const auto& name : input_names) {
            if (obs.find(name) == obs.end()) {
                throw std::runtime_error("Input name " + std::string(name) + " not found in observations.");
            }
        }

        // Create input tensors
        std::vector<Ort::Value> input_tensors;
        for(int i(0); i<input_names.size(); ++i)
        {
            const std::string name_str(input_names[i]);
            auto& input_data = obs.at(name_str);
            if (input_data.size() != input_sizes[i]) {
                throw std::runtime_error("ONNX input vector dimension mismatch");
            }
            for (float value : input_data) {
                if (!std::isfinite(value)) {
                    throw std::runtime_error("ONNX input contains NaN/Inf");
                }
            }
            auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_sizes[i], input_shapes[i].data(), input_shapes[i].size());
            input_tensors.push_back(std::move(input_tensor));
        }

        // Run the model
        auto output_tensor = session->Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_tensors.size(), output_names.data(), 1);

        // Copy output data
        auto floatarr = output_tensor.front().GetTensorMutableData<float>();
        std::lock_guard<std::mutex> lock(act_mtx_);
        std::memcpy(action.data(), floatarr, output_shape[1] * sizeof(float));
        for (float value : action) {
            if (!std::isfinite(value)) {
                throw std::runtime_error("ONNX output contains NaN/Inf");
            }
        }
        return action;
    }

private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<int64_t> input_sizes;
    std::vector<int64_t> output_shape;
};
};
