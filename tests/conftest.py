"""pytest configuration: seed RNGs and report the active device."""
import torch
import pytest


def pytest_configure(config):
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = False   # deterministic for tests
        torch.backends.cudnn.deterministic = True


def pytest_sessionstart(session):
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory // 1024 ** 3
        print(f"\n[device] CUDA — {name} ({mem} GB)")
    else:
        print("\n[device] CPU (no CUDA GPU detected)")
