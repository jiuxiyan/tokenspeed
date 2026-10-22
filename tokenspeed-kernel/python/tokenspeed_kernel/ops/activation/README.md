# Activation kernels

`silu_and_mul` keeps the generic fused SwiGLU contract. The separate
`qwen35_silu_and_mul` entry is used only by Qwen3.5 on Ascend: it evaluates
SiLU on the gate in the input dtype, materializes that result, and then
multiplies by the up projection in the input dtype. This matches the
Transformers 5.12 BF16 Qwen3.5 MLP boundary. Both native fused SwiGLU and
the runtime's generic FP32 expression can otherwise skip that intermediate
rounding and accumulate differences across decoder layers.

The Qwen entry accepts NPU BF16, FP16 and FP32 tensors with any leading
dimensions and an even final width `[gate | up]`, including empty tensors
and strided views. Its optional output has half the final width and the
same dtype/device; the input is preserved unless the output aliases it.
All tensor computation stays on the NPU. It rejects unsupported platforms
and dtypes explicitly. The Qwen runtime module calls this public kernel
boundary; other models and other platforms keep their existing activation.

`qwen35_sigmoid_mul` similarly rounds the Qwen3.5 full-attention sigmoid
gate in the activation dtype before the output product. Generic
`sigmoid_mul` retains its FP32 sigmoid and multiplication contract.
