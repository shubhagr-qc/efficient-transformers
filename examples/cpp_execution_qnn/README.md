
# Text Generation using CPP Inference

## Overview
This example demonstrates how to execute a model on AI 100 using Efficient Transformers and C++ APIs. The Efficient Transformers library is utilized for transforming, exporting and compiling the model, while the QPC is executed using C++ APIs. It is tested on both x86 and ARM platform.

> **_NOTE:_**  This supports BS>1 and Chunking.

## Prerequisite
1. `pip install pybind11`
2. Cpp17 or above (Tested on C++17 and g++ version - 11.4.0)
3. QEfficient [Quick Installation Guide]( https://github.com/quic/efficient-transformers?tab=readme-ov-file#quick-installation)

## Setup and Execution
```bash

# Compile the cpp file using the following commands
mkdir build
cd build

cmake ..
make -j 8

cd ../../../  # Need to be in base folder - efficient-transformers to run below cmd

# Run the python script to get the generated text
python examples/cpp_execution/text_inference_using_cpp.py --model_name gpt2 --batch_size 1 --prompt_len 32 --ctx_len 128 --mxfp6 --num_cores 14 --device_group [0] --prompt "My name is" --mos 1 --aic_enable_depth_first

```

## Future Enhancements
1. DMA Buffer Handling
2. Continuous Batching
3. Handling streamer
