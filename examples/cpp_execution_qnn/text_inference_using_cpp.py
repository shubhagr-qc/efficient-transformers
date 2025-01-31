# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Union

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

import QEfficient
from QEfficient.utils.constants import QnnConstants
from QEfficient.cloud.export import get_onnx_model_path
from QEfficient.generation.text_generation_inference import fix_prompts, get_compilation_dims, get_input_prompts
from QEfficient.utils import check_and_assign_cache_dir, get_qpc_dir_path, load_hf_tokenizer, qpc_exists
from QEfficient.utils.logging_utils import logger
from QEfficient.utils._utils import create_json, execute_command, load_json

import json
import logging
import logging.handlers
import os
import sys
from time import perf_counter
from typing import Dict, List, Optional
import numpy as np
import transformers
import subprocess
import logging
import threading
from huggingface_hub import login


# FORMAT = "%(asctime)s [%(filename).22s:%(lineno).3s] | %(levelname)s | %(message)s"
# logging.basicConfig(
#     level=logging.INFO,
#     format=FORMAT,
#     handlers=[
#         logging.StreamHandler(),
#         logging.FileHandler("runQNN_log.txt")
#     ]
# )
# io_files = []
RUN_KV_MODEL = str(os.path.dirname(os.path.realpath(__file__))) + "/run_kv_model"
def read_stream(stream, prefix, output_list):
    """
    This API will read each line from the stream and print it
    with an appropriate prefix to distinguish between
    standard output and standard error.
    Triggered by run_command_with_output()
    Args:
        stream (IO(str)): Output stream of command
        prefix (str): stderr or stdout prefix
        output_list (list) : appends the stream
    """
    for line in stream:
        logging.info(f"{prefix} {line.rstrip()}")
        output_list.append(line.rstrip())

# script_dir = Path(__file__).resolve().parent
# so_folder_path = script_dir / "build"

# if so_folder_path.is_dir():
#     sys.path.append(str(so_folder_path))
#     try:
#         import InferenceSetIOBuffer  # noqa: E402
#     except ImportError:
#         logger.error("Error importing InferenceSetIOBuffer Module")
#         raise ImportError(
#             "Could not import `InfereceSetIoBuffer` executable, Please refer `examples/cpp_execution/README.md` file for building compiling cpp files."
#         )
# else:
#     raise FileNotFoundError(
#         "Please follow `examples/cpp_execution/README.md` instructions to first compile the cpp files"
#     )


def main(
    model_name: str,
    num_cores: int,
    device_group: Optional[List[int]] = None,
    prompt: Optional[str] = None,  # type: ignore
    prompts_txt_file_path: Optional[str] = None,
    aic_enable_depth_first: bool = False,
    mos: int = -1,
    batch_size: int = 1,
    full_batch_size: Optional[int] = None,
    prompt_len: int = 32,
    ctx_len: int = 128,
    generation_len: Optional[int] = None,
    mxfp6: bool = False,
    mxint8: bool = False,
    local_model_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> None:
    """
    1. Check if compiled qpc for given config already exists, if it does jump to execute, else
    2. Check if exported ONNX file already exists, if true, jump to compilation -> execution, else
    3. Check if HF model exists in cache, if true, start transform -> export -> compilation -> execution, else,
    4. Download HF model -> transform -> export -> compile -> execute
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``gpt2``
        :num_cores (int): Number of cores to compile model on.
    ``Optional`` Args:
        :device_group (List[int]): Device Ids to be used for compilation. If ``len(device_group) > 1``, multiple Card setup is enabled. ``Defaults to None.``
        :prompt (str): Sample prompt for the model text generation. ``Defaults to None.``
        :prompts_txt_file_path (str): Path to txt file for multiple input prompts. ``Defaults to None.``
        :aic_enable_depth_first (bool): Enables ``DFS`` with default memory size. ``Defaults to False.``
        :mos (int): Effort level to reduce the on-chip memory. ``Defaults to -1.``
        :batch_size (int): Batch size to compile the model for. ``Defaults to 1.``
        :full_batch_size (int): Set full batch size to enable continuous batching mode. ``Default to None``
        :prompt_len (int): Prompt length for the model to compile. ``Defaults to 32.``
        :ctx_len (int): Maximum context length to compile the model. ``Defaults to 128.``
        :generation_len (int): Number of tokens to be generated. ``Defaults to False.``
        :mxfp6 (bool): Enable compilation for MXFP6 precision. ``Defaults to False.``
        :mxint8 (bool): Compress Present/Past KV to ``MXINT8`` using ``CustomIO`` config. ``Defaults to False.``
        :local_model_dir (str): Path to custom model weights and config files. ``Defaults to None.``
        :cache_dir (str): Cache dir where downloaded HuggingFace files are stored. ``Defaults to None.``
        :hf_token (str): HuggingFace login token to access private repos. ``Defaults to None.``

    .. code-block:: bash

        python -m examples.text_inference_from_cpp OPTIONS

    """
    cache_dir = check_and_assign_cache_dir(local_model_dir, cache_dir)
    tokenizer = load_hf_tokenizer(
        pretrained_model_name_or_path=(local_model_dir if local_model_dir else model_name),
        cache_dir=cache_dir,
        hf_token=hf_token,
    )

    if full_batch_size is not None:
        raise RuntimeError("Continuous batching will be supported in future, please rerun without continuous batching.")

    qpc_dir_path = get_qpc_dir_path(
        model_name, num_cores, mos, batch_size, prompt_len, ctx_len, mxfp6, mxint8, device_group, full_batch_size, enable_qnn=True,
    )
    if qpc_exists(qpc_dir_path):
        logger.info(f"Pre-compiled qpc found at {qpc_dir_path}! Executing with given prompt")
    else:
        # Handle onnx model generation
        onnx_model_path = get_onnx_model_path(
            model_name, cache_dir, tokenizer, hf_token, local_model_dir, full_batch_size
        )
        _ = QEfficient.compile(
            onnx_path=onnx_model_path,
            qpc_path=os.path.dirname(
                qpc_dir_path
            ),  # We need to pass parent directory of qpc_dir_path, as the compile function handles the qpcs directory creation
            num_cores=num_cores,
            batch_size=batch_size,
            prompt_len=prompt_len,
            ctx_len=ctx_len,
            mxfp6=mxfp6,
            mxint8=mxint8,
            aic_enable_depth_first=aic_enable_depth_first,
            mos=mos,
            device_group=device_group,
            full_batch_size=full_batch_size,
            enable_qnn=True,
        )


    context_binary_path = os.path.join(qpc_dir_path, f"{QnnConstants.CONTEXT_BIN_NAME}.bin")

    #########
    # Execute
    #########
    execute_qnn_binary_cpp(
        tokenizer=tokenizer,
        qpc=context_binary_path,
        prompt=prompt,
        batch_size=batch_size,
        ctx_len=ctx_len,
        generation_len=generation_len,
        device_ids=device_group,
        seq_len=prompt_len,
    )

def execute_qnn_binary_cpp(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    qpc: str,
    prompt: List[str],
    batch_size: int,
    ctx_len: int,
    generation_len: Optional[int] = None,
    seq_len: int = 128,
    device_ids: list = [],
    profiling: bool = False
):
    print(qpc)
    manual_devices = "0"
    # device_ids
    if len(device_ids) > 0:
        devices = " ".join(str(device_ids))
        manual_devices = f"{len(device_ids)} {devices}".strip()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    max_id = max(tokenizer.vocab.values())
    id_to_token = {v: k for k, v in tokenizer.vocab.items()}
    vocab_size = len(id_to_token)
    assert set(id_to_token) == set(range(vocab_size)), "Missing tokens"
    with open("tokens.bin", "wb") as fp:
        fp.write(str(vocab_size).encode())
        fp.write(b"\0")
        for i in range(vocab_size):
            token = id_to_token[i]
            if token[0] == "Ġ":
                token = " " + token[1:]
            fp.write(token.encode())
            fp.write(b"\0")
    # Read prompt, zero-th index and ctx len from session
    prefill_seq_len: int = seq_len
    # TODO: Placeholder code for batchsize and chunking -- NOT USED
    # expanding prompt array to match batch size
    if len(prompt) < batch_size:
        prompt = prompt * -(batch_size // -len(prompt))  # Repeat prompt to required size
    prompt = prompt[:batch_size]  # Truncate prompts to required size
    # if user didn't provide input_len, obtain input_len from tokenizer
    inputs = tokenizer(prompt, return_tensors="np", padding=True)
    input_len = inputs["attention_mask"].sum(1, keepdims=False)[0]
    padded_len = inputs["input_ids"].shape[1]
    num_chunks = -(padded_len // -prefill_seq_len)  # ceil divide without float
    padded_input_len = num_chunks * prefill_seq_len  # Convert to a multiple of prompt_len
    print('num_chunks: ', num_chunks)
    print('input_len', input_len)
    print('padded_len', padded_len)
    print('prompt_len(seq_len): ', prefill_seq_len)
    print('num_chunk * prompt_len ', padded_input_len)
    print(generation_len)
    if generation_len is None or generation_len == 0:
        generation_len = ctx_len - padded_input_len
    print(generation_len)

    assert generation_len > 0, "generation length should be greater than zero"
    # TODO: for now, use single prompt
    prompt_0: str = prompt[0]
    # Prepare inputs for first iteration
    inputs = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_input_len)
    inputs["position_ids"] = np.where(inputs.pop("attention_mask"), np.arange(padded_input_len), -1)
    # Need to use -1 as position_ids for invalid tokens
    # generated_ids = np.full((batch_size, generation_len + 1), tokenizer.pad_token_id)
    chunks = []
    for i in range(num_chunks):
        chunk_inputs = inputs.copy()
        chunk_inputs["input_ids"] = inputs["input_ids"][:, i * prefill_seq_len: (i + 1) * prefill_seq_len]
        chunk_inputs["position_ids"] = inputs["position_ids"][:, i * prefill_seq_len: (i + 1) * prefill_seq_len]
        chunks.append(chunk_inputs)
    os.makedirs("prefill", exist_ok=True)
    for index, chunk_inputs in enumerate(chunks):
        for key, value in chunk_inputs.items():
            chunk_inputs[key].tofile(f"prefill/{key}_{index}.raw")
    with open("prefill/input_file.txt", 'w') as fd:
        for index, chunk_inputs in enumerate(chunks):
            fd.write(",".join([f"prefill/{input}_{index}.raw" for input in (list(chunk_inputs.keys()))]) + '\n')
    profiling_flag = 1 if profiling else 0
    run_command = f"{RUN_KV_MODEL} {qpc} \"{prompt_0}\" {num_chunks} {input_len} {ctx_len} {generation_len} {profiling_flag} {manual_devices} {tokenizer.eos_token_id}"

    execute_command("CPP_RUNTIME", run_command, os.getcwd())


def cloud_ai_100_exec_kv_cpp(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    qpc_path: str,
    prompt_len: int,
    prompt: Optional[List[str]] = None,
    prompts_txt_file_path: Optional[str] = None,
    device_id: Optional[List[int]] = None,
    generation_len: Optional[int] = None,
    enable_debug_logs: bool = False,
    stream: bool = True,
    full_batch_size: Optional[int] = None,
):
    batch_size, ctx_len = get_compilation_dims(qpc_path)
    prompt: List[str] = get_input_prompts(prompt, prompts_txt_file_path)
    prompt = fix_prompts(prompt, batch_size, full_batch_size)

    # ********* CPP Calling ********
    InferenceSetIOBuffer.generatePrompt(
        tokenizer, qpc_path, prompt_len, ctx_len, batch_size, prompt, generation_len, device_id
    )


def tokenize_for_prefill(prompt, tokenizer):
    inputs = tokenizer(prompt, return_tensors="np", padding=True)
    return inputs


def tokenize_for_prefill_with_padded_len(prompt, tokenizer, padded_len):
    inputs = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
    return inputs


# Generating text from generated_ids received from Python
def tokenize_decode_output(tokenizer, generated_ids, prompt):
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    for g, p in zip(generated_texts, prompt):
        print("Prompt: ", p)
        print("Generated Text: ", g)
        print()
    return generated_texts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference command, the model will be downloaded from HF, optimized, compiled, executed on Cloud AI 100"
    )
    parser.add_argument("--model-name", "--model_name", required=True, help="HF Model card name/id")
    parser.add_argument(
        "--local-model-dir", "--local_model_dir", required=False, help="Path to custom model weights and config files"
    )
    parser.add_argument(
        "--cache-dir",
        "--cache_dir",
        default=None,
        required=False,
        help="Cache dir to store HF Downloads",
    )
    parser.add_argument(
        "--hf-token", "--hf_token", default=None, type=str, required=False, help="HF token id for private HF models"
    )
    parser.add_argument("--batch-size", "--batch_size", type=int, default=1, help="Batch size for text generation")
    parser.add_argument(
        "--prompt-len", "--prompt_len", default=32, type=int, help="Sequence length for text generation."
    )
    parser.add_argument("--ctx-len", "--ctx_len", default=128, type=int, help="Context length for text generation.")
    parser.add_argument(
        "--mxfp6", action="store_true", help="Compress constant MatMul weights to MXFP6 E2M3, default is no compression"
    )
    parser.add_argument(
        "--mxint8",
        action="store_true",
        help="Compress Present/Past KV to MXINT8 using CustomIO config, default is False",
    )
    parser.add_argument(
        "--num_cores", "--num-cores", type=int, required=True, help="Number of cores to compile on Cloud AI 100"
    )
    parser.add_argument(
        "--device_group",
        "--device-group",
        type=lambda device_ids: [int(x) for x in device_ids.strip("[]").split(",")],
        help="Cloud AI 100 device ids (comma-separated) e.g. [0,1]  ",
    )
    parser.add_argument(
        "--prompt",
        type=lambda prompt: prompt.split("|"),
        help="Input prompt, if executing for batch size>1, pass input prompts in single string but separate with pipe (|) symbol",
    )
    parser.add_argument(
        "--prompts_txt_file_path",
        "--prompts-txt-file-path",
        type=str,
        help="File path for taking input prompts from txt file, sample prompts.txt file present in examples folder",
    )
    parser.add_argument("--generation_len", "--generation-len", type=int, help="Number of tokens to generate")
    parser.add_argument(
        "--aic_enable_depth_first",
        "--aic-enable-depth-first",
        action="store_true",
        help="If passed, this option will be enabled during compilation, disabled by default",
    )
    parser.add_argument(
        "--mos",
        type=int,
        default=-1,
        help="Effort level to reduce the on-chip memory",
    )
    # FIXME: Add verbose feature
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="pass to print info logs",
    )

    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.INFO)
    del args.verbose  # type: ignore
    main(**args.__dict__)
