# Causal convolution

The portable Triton prefill and decode kernels retain the raw projected inputs
in the rolling state. For a width-four kernel the state holds the last three
inputs, independently of activation and output precision.

Ascend prefill uses a 64-channel tile to fit the device unified buffer; other
platforms retain the existing 256-channel tile. All channels are evaluated.

The Ascend text baseline follows the Transformers portable Qwen3.5 precision
boundaries: accumulate the convolution in FP32, round to the input dtype, apply
SiLU in FP32, then store in the input dtype. The intermediate rounding matters
for BF16 Q/K normalization and subsequent recurrent updates. Other platforms
retain their existing fused accumulation-through-SiLU behavior. The explicit
`ROUND_BEFORE_ACTIVATION` specialization is fixed by the device platform, not
by request lengths. It does not alter the values retained in the rolling state.

The Ascend tests in `tokenspeed-kernel-npu/test/test_causal_conv_task1.py`
compare a CPU per-tap reference and exact state copies for the model's 6144
channels, window boundaries, nonzero initial states and repeated decode.
