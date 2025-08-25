# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

"""
This script demonstrates the end-to-end workflow of using the QAIRT library for
model conversion, loading, compilation, and execution with the AIC backend.

It also showcases optional I/O, network specialization, and graph selection
features for the Llama-3.1-8B model.
"""

import json
import sys
import tempfile
from pathlib import Path, PurePath
import os
import numpy as np

try:
    import qairt
except ImportError:
    import platform
    import sys

    sys.path.append(f"{os.getenv('QNN_SDK_ROOT')}lib/python/qairt/")
    import qairt

from qairt import CompileConfig
from qairt.api.compiler.config import AicCompilerConfig
'''
# Define paths for the model and I/O configuration
# IMPORTANT: Update these paths to point to your actual model and config files.
MODEL_PATH = PurePath(
    "/workspace/qeffHome/LlamaForCausalLM/LlamaForCausalLM-bb1d70c8e81d2504/LlamaForCausalLM.onnx"
)
CUSTOM_IO_CONFIG_PATH = Path(__file__).parent / "Llama_3_1_8B_io_config_optional_inputs.yaml"
OUTPUT_DLC_PATH = Path("model.dlc")

print("Starting QAIRT End-to-End Workflow...")

# --- Step 1: Model Conversion ---
print(f"\n--- Step 1: Converting model from {MODEL_PATH} ---")
converted_model = qairt.convert(
    MODEL_PATH,
    io_config=CUSTOM_IO_CONFIG_PATH,
    float_precision=16,
    float_bias_precision=32,
    preserve_io_datatype="all",
    defer_loading=True,
    onnx_simplification=False,
)

# Save the converted model to a .dlc file
converted_model.save(str(OUTPUT_DLC_PATH))
print(f"Converted specialized DLC model saved to: {OUTPUT_DLC_PATH}")

print("\nContinuing with model loading, compilation, and execution...")

# Load the DLC model
loaded_model = qairt.load(OUTPUT_DLC_PATH)

# --- Step 2: Model Compilation ---
print("\n--- Step 2: Compiling model for AIC ---")
# Define compiler configuration for AIC backend
aic_compiler_config = AicCompilerConfig(
    graph_names=["LlamaForCausalLM_configuration_2", "LlamaForCausalLM_configuration_1"],
    compilation_target="hardware",
    hardware_version="2.0",
    num_of_cores=14,
    stat_level=40,
    stats_batch_size=1,
    print_perf_metrics=True,
    perf_warnings=True,
    pmu_recipe_opt="KernelUtil",
    enable_depth_first=True,
    max_out_channel_split="1",
    overlap_split_factor=2,
    size_split_granularity=1024,
    do_ddr_to_multicast=True,
    enable_debug=True,
    retained_state=True,
)

compiled_model = qairt.compile(
    model=loaded_model,
    config=CompileConfig(
        backend="AIC",
        debug=False,
        compiler_custom_configs=aic_compiler_config,
    ),
)
print("Model compiled successfully!")
'''
# --- Prepare Input Data for Inference from dummy data---
# This structure assumes the model has two configurations/paths for Llama-3.1-8B
# requiring different input shapes/types.
inp_type_dict = [
    {  # Input for the first configuration
        "batch_index": np.zeros((1, 1), dtype=np.int64),
        "input_ids": np.zeros((1, 128), dtype=np.int64),
        "position_ids": np.zeros((1, 128), dtype=np.int64),
    },
    # {  # Input for the second configuration
    #     "batch_index": np.zeros((64, 1), dtype=np.int64),
    #     "input_ids": np.zeros((64, 1), dtype=np.int64),
    #     "position_ids": np.zeros((64, 1), dtype=np.int64),
    # },
]

compiled_model = qairt.load("/workspace/qeffHome/LlamaForCausalLM/LlamaForCausalLM-bb1d70c8e81d2504/qpc-fa5b47b38cceef61/qnngraph.serialized.bin")
# --- Step 3: Model Execution (Local x86 with AIC backend simulation) ---

print("\n--- Step 3: Executing compiled model locally (AIC backend simulation) ---")
exec_result_local = compiled_model(
    inputs=inp_type_dict,
    backend="AIC",
    num_inferences=1,
    use_native_input_data=True,
    profiling_level="basic",
    log_level="debug",
    graph_names=["model_configuration_1"],
)

print("\nModel execution complete. Output:")
# Print the output from the execution
assert isinstance(exec_result_local.data, dict)
for outer_key, inner_dict in exec_result_local.data.items():
    print(f"\n")
    print(f"Graph Name: '{outer_key}'")
    assert isinstance(inner_dict, dict)
    for inner_key, ndarray_value in inner_dict.items():
        print(f"\n")
        print(f"    Output Name: '{inner_key}'")
        print(f"        Type: {type(ndarray_value)}")
        print(f"        Dtype: {ndarray_value.dtype}")
        print(f"        Shape: {ndarray_value.shape}")
        print(f"        Size: {ndarray_value.size}")

'''
# --- Step 4: Save Compiled Model and Info ---
print("\n--- Step 4: Saving compiled model and info ---")
# Define output directory for compiled artifacts
output_dir = Path("aic_outputs")
output_dir.mkdir(exist_ok=True)

compiled_bin_path = output_dir / "compiled_model.bin"
compiled_info_json_path = output_dir / "compiled_model_info.json"

try:
    # Save compiled model as context binary file (.bin)
    compiled_model.save(str(compiled_bin_path))
    print(f"Compiled model saved to: {compiled_bin_path}")

    # Save binary info as JSON file
    with open(compiled_info_json_path, "w") as fp:
        json.dump(compiled_model.module.info.as_dict(), fp, indent=4)
    print(f"Compiled model information saved to: {compiled_info_json_path}")

except Exception as e:
    print(f"Error while saving compiled model artifacts: {e}")
    sys.exit(1)
'''
print("\n--- Script finished successfully! ---")