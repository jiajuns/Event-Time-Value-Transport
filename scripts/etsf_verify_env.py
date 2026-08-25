"""Verify the isolated RoboTwin environment on the remote host."""

import importlib


MODULES = [
    "torch",
    "torchvision",
    "numpy",
    "scipy",
    "sapien",
    "mplib",
    "pytorch3d",
    "curobo",
    "warp",
    "xpolicylab",
]


for name in MODULES:
    try:
        module = importlib.import_module(name)
        print(
            name,
            getattr(module, "__version__", "present"),
            getattr(module, "__file__", ""),
        )
    except Exception as exc:  # noqa: BLE001 - diagnostic script reports all imports.
        print(name, "ERROR", type(exc).__name__, str(exc))

import torch

print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("cuda_device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
cuda_tensor = torch.arange(5, device="cuda")
print("cuda_tensor_sum", cuda_tensor.sum().item())

try:
    from pytorch3d import _C  # noqa: F401

    print("pytorch3d_C", "OK")
except Exception as exc:  # noqa: BLE001 - diagnostic script reports extension failures.
    print("pytorch3d_C", "ERROR", type(exc).__name__, str(exc))
