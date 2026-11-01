"""Проверить установленные версии и доступность GPU."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import ROOT, write_json
from importlib.metadata import version
import json
import torch

result = {"python": sys.version.split()[0]}
for name in ["torch", "numpy", "transformers", "tokenizers", "tensorboard", "matplotlib"]:
    result[name] = version(name)
result["cuda_available"] = torch.cuda.is_available()
result["bf16_supported"] = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
if torch.cuda.is_available():
    result["gpu"] = torch.cuda.get_device_name()
    result["flash_sdp_built"] = torch.backends.cuda.is_flash_attention_available()
print(json.dumps(result, ensure_ascii=False, indent=2))
write_json(ROOT / "reports/environment.json", result)
