import torch
from transformers.onnx import OnnxConfigWithPast
from collections import OrderedDict

class LlamaOnnxConfig(OnnxConfigWithPast):
    def __init__(self, config, task="default"):
        super().__init__(config, task=task)

    @property
    def inputs(self):
        inputs = OrderedDict([
            ("input_ids", {0: "batch", 1: "sequence"}),
        ])
        if self.use_attention_mask:
            inputs["attention_mask"] = {0: "batch", 1: "sequence"}
        if self.use_past:
            for i in range(self.num_layers):
                inputs[f"past_key_values.{i}.key"] = {0: "batch", 2: "past_sequence"}
                inputs[f"past_key_values.{i}.value"] = {0: "batch", 2: "past_sequence"}
        return inputs

    @property
    def outputs(self):
        outputs = OrderedDict([
            ("logits", {0: "batch", 1: "sequence"}),
        ])
        if self.use_past:
            for i in range(self.num_layers):
                outputs[f"present.{i}.key"] = {0: "batch", 2: "past_sequence + sequence"}
                outputs[f"present.{i}.value"] = {0: "batch", 2: "past_sequence + sequence"}
        return outputs

    def generate_dummy_inputs(self, tokenizer, batch_size=1, seq_length=1):
        inputs = {
            "input_ids": torch.randint(0, self._config.vocab_size, (batch_size, seq_length), dtype=torch.long)
        }
        if self.use_attention_mask:
            inputs["attention_mask"] = torch.ones((batch_size, seq_length), dtype=torch.long)
        if self.use_past:
            inputs["past_key_values"] = [
                (
                    torch.zeros((batch_size, self.num_attention_heads, 0, self.head_dim)),
                    torch.zeros((batch_size, self.num_attention_heads, 0, self.head_dim))
                ) for _ in range(self.num_layers)
            ]
        return inputs

    @property
    def use_past(self):
        return False  # 如果需要使用 past_key_values，请设置为 True

    @property
    def use_attention_mask(self):
        return True

    @property
    def num_layers(self):
        return self._config.num_hidden_layers

    @property
    def num_attention_heads(self):
        return self._config.num_attention_heads

    @property
    def head_dim(self):
        return self._config.hidden_size // self._config.num_attention_heads