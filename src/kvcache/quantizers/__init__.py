"""量化器：我们的方法 + KIVI / KVQuant baseline。

粒度：per_channel / per_token / per_head / per_layer
位数：1/2/4-bit 扫描（配置见 configs/quantization/*.yaml）

基准对照：
- KIVI:     2-bit 分组量化，组大小 16
- KVQuant:  感知校准的量化，含 per-channel 离群处理
"""
