import torch

print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    # 做一次小计算，确认真的在 GPU 上跑
    x = torch.randn(2048, 2048, device="cuda")
    y = x @ x
    print("gpu_matmul_mean:", y.mean().item())

