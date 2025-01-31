# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
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
README = """
Commands -
python3 runKVCacheSingleModel.py --model-name facebook/opt-125m --qpc opt-125m-kv-context-binary_working.bin --batch-size 1 --ctx-len 256 --seq-len 128 --generation-len 128 --profiling --prompt "Thanks for being here"
Capabilities
- prefill, decode, and total tok/s will be generated
- Chunking support added
- Batchsize support is added
- Device selection is added
- Decode only mode is supported (seq_len 1, single graph)
Limitations (for parity)
 - Streaming is disabled, so result will be printed after the run is completed.
 - batch_size: int = 1, prompt_len/seq_len: int = 128, ctx_len: int = 2048, need to pass these as flags for now
   (these info are obtained in AIC, from the binary / input)
 - Full memory allocation of happens in each run
 example :-
 prompt's len - 10 (eg:- "my name is")
 input_len - 1024
 seq_len = 128
 ctx_len = 2048
 gen_len = 1024
 old approach
 |<--tokenizer.padding->|<--input-->|<p>|<----------gen_len----------->|
 0                     896         906 1024                            2048
 new approach
 |<--input-->|<->|<----------gen_len----------->|
 0           10 128                         1024+128
"""

FORMAT = "%(asctime)s [%(filename).22s:%(lineno).3s] | %(levelname)s | %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("runQNN_log.txt")
    ]
)
io_files = []
RUN_KV_MODEL = str(os.path.dirname(os.path.realpath(__file__))) + "/run_kv_model"
RUN_KV_MODEL_JAIS = str(os.path.dirname(os.path.realpath(__file__))) + "/run_kv_model_jais"
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
def run_command(command, timeout=60000) -> tuple:
    """
    This API will execute the shell command,
    and collect output stdout and stderr
    Args:
        command (str): Command that needed to be executed
        timeout: Timeout
    Returns:
        stdout_list (list): stdout list
        stderr_list (list): stderr list
        return_code (int): returns the command status code
    """
    print(f"Executing \n {command}")
    stdout_list = []
    stderr_list = []
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            shell=True,
        )
        msg = ""
        stop_event = threading.Event()
        stdout_thread = threading.Thread(
            target=read_stream,
            args=(process.stdout,
                  f"{msg}[stdout] ",
                  stdout_list)
        )
        stderr_thread = threading.Thread(
            target=read_stream,
            args=(process.stderr,
                  f"{msg}[stderr] ",
                  stderr_list)
        )
        stdout_thread.start()
        stderr_thread.start()
        # Wait for the process to complete
        process.wait(timeout=timeout)
        # Wait for the threads to finish
        stop_event.set()
        stdout_thread.join()
        stderr_thread.join()
        # Get the return code of the process
        return_code = process.returncode
    except subprocess.TimeoutExpired:
        logging.error(f"Timeout expired :\n {command} FAILED")
        process = None
        return_code = -1
        stderr_list.append(f"Timeout expired :\n {command} FAILED")
        stdout_list.append("")
    return stdout_list, stderr_list, return_code
def main(
    model_name: str,
    qpc: str,
    prompt: List[str],
    batch_size: int,
    ctx_len: int,
    generation_len: Optional[int] = None,
    # stream: bool = True,
    # enable_debug_logs: bool = False,
    # write_io_dir: Optional[str] = None,
    seq_len: int = 128,
    device_ids: list = [],
    profiling: bool = False
):
    is_jais = False  # enable if token_type_ids is present
    manual_devices = "0"
    # device_ids
    if len(device_ids) > 0:
        devices = " ".join(device_ids)
        manual_devices = f"{len(device_ids)} {devices}".strip()
    login(token=HF_TOKEN)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, padding_side="right", trust_remote_code=True
    )
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
    if generation_len is None:
        generation_len = ctx_len - padded_input_len
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
    run_kv = RUN_KV_MODEL_JAIS if is_jais else RUN_KV_MODEL  # picking the right script (jais or no jais)
    profiling_flag = 1 if profiling else 0
    stdout_list, stderr_list, return_code = \
        run_command(
            f"{run_kv} {qpc} \"{prompt_0}\" {num_chunks} {input_len} {ctx_len} {generation_len} {profiling_flag} {manual_devices} {tokenizer.eos_token_id}"
        )
    input_prompt = ""
    output_string = ""
    result = ""
    log_list = []
    for stdout in stdout_list:
        if "[RESULT]" in stdout:
            result = stdout
        elif "[INPUT]" in stdout:
            input_prompt = stdout
        elif "[OUTPUT]" in stdout:
            output_string = stdout
        elif "[LOG]" in stdout:
            log_list.append(stdout.replace("[LOG]", "").strip())
        else:
            logging.debug(stdout)
    input_prompt = input_prompt.replace("[INPUT]", "").strip()
    output_string = output_string.replace("[OUTPUT]", "").strip()
    result = result.replace("[RESULT]", "").strip()
    for stderr in stderr_list:
        logging.error(stderr)
    try:
        result = json.loads(result)
    except json.JSONDecodeError:
        logging.info("Result generated is not in proper format, exiting.")
        sys.exit(1)
    before_prefill_time = 0.0
    total_prefill_time = 0.0
    decode_start_time = 0.0
    total_decode_time = 0.0
    for log in log_list:
        try:
            log_data = json.loads(log)
        except json.JSONDecodeError:
            continue
        except Exception as e:
            logging.error("Exception while processing log data - " + str(e))
        if log_data.get("message", "").strip() == "Starting prefill":
            before_prefill_time = log_data['timestamp']
        elif log_data.get("message", "").strip() == "Prefill completed":
            total_prefill_time = log_data['timestamp'] - before_prefill_time
        elif log_data.get("message", "").strip() == "Decode begin":
            decode_start_time = log_data['timestamp']
        elif log_data.get("message", "").strip() == "Decode stage complete":
            total_decode_time = log_data['timestamp'] - decode_start_time
        else:
            logging.debug(log)
    logging.info(f"returncode : {return_code}")
    # logging.info(f"result : {result}")
    print(f"input= [{input_prompt}]")
    print(f"output= [{output_string}]")
    try:
        print("Prefill token/sec is=", round(float(result["prefill_tps"]) * batch_size, 2))
        print("Decode token/sec is=", round(float(result["decode_tps"]) * batch_size, 2))
        print("Total token/sec is=", round(float(result["total_tps"]) * batch_size, 2))
        print("Time taken before prefill stage in secs=", round(before_prefill_time, 4))
        print("Time taken at prefill stage in secs=", round(total_prefill_time, 4))
        print("Time taken at decode stage in secs=", round(total_decode_time, 4))
        print("Time taken overall in secs=", round(before_prefill_time + total_prefill_time + total_decode_time, 4))
    except TypeError:
        logging.error("Run failed, result generated is not in proper format, exiting..")
if __name__ == "__main__":
    import argparse
    argp = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    argp.description = README
    argp.add_argument("--model-name", required=True, help="Model name to run")
    argp.add_argument("--qpc", required=True, help="Compiled binary QPC/ QNN bin")
    argp.add_argument(
        "--prompt",
        type=lambda prompt: prompt.split("|"),
        default="My name is",
        help="Input prompt(s) to generate for (pipe-separated)",
    )
    argp.add_argument("--batch-size", default=1, type=int, help="Batch size")
    argp.add_argument("--ctx-len", required=True, type=int, help="Context length")
    argp.add_argument("--seq-len", default=128, type=int, help="Seq length of the model")
    argp.add_argument(
        "--device_ids",
        default=[],
        type=lambda devices: [device.strip() for device in devices.split(',')],
        help="Comma seperated device ids. ie., 0,1,2,3"
    )
    argp.add_argument(
        "--profiling",
        action="store_true",
        help="Enable generating profiling"
    )
    argp.add_argument(
        "--generation-len",
        type=int,
        required=True,
        help="Number of tokens to generate. \
              Note: For models without rolling buffer, (generation length + input length) should \
              be less than model context length",
    )
    # argp.add_argument(
    #     "--no-stream", action="store_false", dest="stream", help="Don't stream output text"
    # )
    # argp.add_argument("--enable-debug-logs", action="store_true", help="Enable debug logs in LRT")
    # argp.add_argument("--write-io-dir", help="Directory to write inputs/outputs into")
    # argp.add_argument(
    #     "--automation", action="store_true", help="Print outputs in required format for automation"
    # )
    args = argp.parse_args()
    main(**vars(args))
