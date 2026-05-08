"""
Phase 2.2 — submit Qwen3-4B-shape attention block to AI Hub.

Submit-only (no polling) so we don't block; check_job() afterward.
"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(__file__))
from attention_block import build_attention_block_onnx
import qai_hub as hub
import onnx

m = build_attention_block_onnx(seq_len=64, d_model=2048, n_heads=16, head_dim=128)
with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
    onnx.save(m, f.name)
    p = f.name
print(f'ONNX: {p} ({os.path.getsize(p)/1024/1024:.1f} MB)')

dev = next(d for d in hub.get_devices() if d.name == 'Samsung Galaxy S22 Ultra 5G')
print('submitting compile (upload may take 1-2 min)...')
cj = hub.submit_compile_job(
    model=p, device=dev,
    name='attn_qwen3_4b_compile',
    options='--target_runtime qnn_context_binary',
)
print('compile job:', cj.job_id)
print('url:', cj.url)
