from QEfficient import QEFFAutoModelForCausalLM
from transformers import AutoTokenizer
from QEfficient.utils.logging_utils import logger
import logging

logger.setLevel(logging.INFO)
model_name = "meta-llama/Llama-3.1-8B-Instruct"
model = QEFFAutoModelForCausalLM.from_pretrained(model_name)
onnx_path =  model.export()
print(onnx_path)
model.compile(
    prefill_seq_len = 128,
    ctx_len = 256,
    enable_qnn=True,
    qnn_config="QEfficient/compile/qnn_config.json")
model.generate(prompts=["Once upon a time"],
               tokenizer = AutoTokenizer.from_pretrained(model_name),
               device_id=[4],
               generation_len=1,
               enable_qnn_runtime=True,
               )
