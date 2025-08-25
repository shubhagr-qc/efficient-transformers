# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from QEfficient.utils.constants import QnnConstants

try:
    import qairt
except ImportError:
    import sys

    sys.path.append(f"{os.getenv('QNN_SDK_ROOT')}lib/python/qairt/")
    import qairt

from qairt.api.common.backends.aic import AicRuntimeConfig

qnn_to_np_dtype_mapping = {
    "QNN_DATATYPE_INT_8": np.dtype(np.int8),
    "QNN_DATATYPE_INT_16": np.dtype(np.int16),
    "QNN_DATATYPE_INT_32": np.dtype(np.int32),
    "QNN_DATATYPE_INT_64": np.dtype(np.int64),
    "QNN_DATATYPE_UINT_8": np.dtype(np.uint8),
    "QNN_DATATYPE_UINT_16": np.dtype(np.uint16),
    "QNN_DATATYPE_UINT_32": np.dtype(np.uint32),
    "QNN_DATATYPE_UINT_64": np.dtype(np.uint64),
    "QNN_DATATYPE_FLOAT_16": np.dtype(np.float16),
    "QNN_DATATYPE_FLOAT_32": np.dtype(np.float32),
}


class QnnAicInferenceSession:
    def __init__(
        self,
        bin_path: Union[Path, str],
        device_ids: Optional[List[int]] = None,
        enable_debug_logs: bool = False,
    ):
        """
        Initialise for QNN-AIC inference Session
        ---------

        :bin_path: str. Path to the save generated binary file after compilation.
        :device_ids: List[int]. Device Ids to be used for compilation. if devices > 1, it enables multiple card setup.
        :activate: bool. If false, activation will be disabled. Default=True.
        :enable_debug_logs: bool. If True, It will enable debug logs. Default=False.
        """

        # Load QNN Compiler binary
        bin_path = os.path.abspath(f"{bin_path}/{QnnConstants.CONTEXT_BIN_NAME}.bin")
        self.compiled_model = qairt.load(bin_path)

        self.log_level = "debug" if enable_debug_logs else "error"
        self.device_ids = device_ids

        # Extract Graph Names
        graphs_info = self.compiled_model.graphs_info

        self.graphs_info = []

        for graph in graphs_info:
            graph_name = graph.name
            input_dict = {}
            output_dict = {}
            for input in graph.inputs:
                input_dict[input.name] = {
                    "dimensions": input.dimensions,
                    "data_type": qnn_to_np_dtype_mapping[input.data_type],
                }
            for output in graph.outputs:
                output_dict[output.name] = {
                    "dimensions": output.dimensions,
                    "data_type": qnn_to_np_dtype_mapping[output.data_type],
                }
            self.graphs_info.append({"name": graph_name, "input": input_dict, "output": output_dict})

        self.inputs_name = list(self.graphs_info[0]["input"].keys())
        self.outputs_name = list(self.graphs_info[0]["output"].keys())
        self.vocab_size = self.graphs_info[0]["output"]["logits"]["dimensions"][2]
        self.full_batch_size = self._fetch_full_batch_size()
        self.decode_seq_len = self._fetch_decode_seq_len()
        self.batch_size, self.prefill_seq_len = self._fetch_batch_size_prefill_seq_len()

    @property
    def input_names(self) -> List[str]:
        return self.inputs_name

    @property
    def output_names(self) -> List[str]:
        return self.outputs_name

    def activate(self):
        """QNN Run time APIs doesn't require activation"""
        return

    def deactivate(self):
        """QNN Run time APIs doesn't require deactivation"""
        return

    def set_buffers(self, buffers: Dict[str, np.ndarray]):
        """QNN Run time APIs doesn't require setting buffers"""
        return

    def skip_buffers(self, skipped_buffer_names: List[str]):
        """QNN Run time APIs doesn't require skipping buffers"""
        return

    def _fetch_full_batch_size(self):
        """Compute the full batch size"""
        full_batch_size = None
        if "batch_index" in self.inputs_name:
            max([graph_info["input"]["batch_index"]["dimensions"][0] for graph_info in self.graphs_info])
        return full_batch_size

    def _fetch_decode_seq_len(self):
        """Compute the decode seq length"""
        return min([graph_info["input"]["input_ids"]["dimensions"][1] for graph_info in self.graphs_info])

    def _fetch_batch_size_prefill_seq_len(self):
        """Fetch the batch size and prefill sequence length"""
        # batch_size
        batch_size = max([graph_info["input"]["input_ids"]["dimensions"][0] for graph_info in self.graphs_info])
        # prefill_seq_len
        prefill_seq_len = max([graph_info["input"]["input_ids"]["dimensions"][1] for graph_info in self.graphs_info])

        return batch_size, prefill_seq_len

    def validate_inputs(self, inputs: Dict[str, np.ndarray]) -> str:
        """
        Validate the input data and return the graph name
        ---------
        :inputs: Dict[str, np.ndarray]. Dictionary of input names and numpy input data.
        """
        graph_name = ""
        for graph_info in self.graphs_info:
            graph_name = graph_info["name"]
            for input_name in inputs.keys():
                if input_name not in self.inputs_name:
                    raise ValueError(f"Input {input_name} is not valid")

                if list(inputs[input_name].shape) != graph_info["input"][input_name]["dimensions"]:
                    graph_name = ""
                    break

            if graph_name != "":
                break
        if graph_name == "":
            raise ValueError(f"Invalid Input: Dimension is not matching with any graph {inputs}")
        return graph_name

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Execute on cloud AI 100

        Args:
            :inputs (Dict[str, np.ndarray]): Processed numpy inputs for the model.

        Return:
            :Dict[str, np.ndarray]:
        """
        graph_name = self.validate_inputs(inputs)
        inputs = [dict(inputs)]

        model_output = self.compiled_model(
            inputs=inputs,
            backend="AIC",
            num_inferences=1,
            use_native_input_data=True,
            profiling_level="basic",
            log_level=self.log_level,
            graph_names=[graph_name],
            runtime_custom_config=AicRuntimeConfig(device_ids=self.device_ids),
        )

        if not isinstance(model_output.data, dict):
            raise RuntimeError("Failed to run inference")

        output = {"logits": model_output.data[graph_name]["logits"]}
        return output
