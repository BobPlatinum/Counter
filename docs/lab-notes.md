# Lab Notes

## `nvidia-smi` Output

```powershell
(base) PS D:\Coding\PycharmCode\Counter> nvidia-smi
Sat Jan 31 02:56:47 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 591.74                 Driver Version: 591.74         CUDA Version: 13.1     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4070 ...  WDDM  |   00000000:01:00.0 Off |                  N/A |
| N/A   42C    P8              2W /  120W |       0MiB /   8188MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            8932    C+G   ...ram Files\Tencent\QQNT\QQ.exe      N/A      |
|    0   N/A  N/A           32500    C+G   ...ram Files\Tencent\QQNT\QQ.exe      N/A      |
+-----------------------------------------------------------------------------------------+
```

## `check_gpu.py` Output

```
torch: 2.7.1+cu118
torch.version.cuda: 11.8
cuda_available: True
gpu: NVIDIA GeForce RTX 4070 Laptop GPU
gpu_matmul_mean: -0.015605181455612183
```

## `yolo checks` Output

```
Ultralytics 8.4.9  Python-3.11.14 torch-2.7.1+cu118 CUDA:0 (NVIDIA GeForce RTX 4070 Laptop GPU, 8188MiB)
Setup complete  (32 CPUs, 31.6 GB RAM, 390.6/1091.5 GB disk)

OS                     Windows-10-10.0.26200-SP0
Environment            Windows
Python                 3.11.14
Install                pip
Path                   D:\Anaconda\envs\HumanCounter3\Lib\site-packages\ultralytics
RAM                    31.63 GB
Disk                   390.6/1091.5 GB
CPU                    13th Gen Intel Core i9-13980HX
CPU count              32
GPU                    NVIDIA GeForce RTX 4070 Laptop GPU, 8188MiB
GPU count              1
CUDA                   11.8

numpy                   2.4.1>=1.23.0
matplotlib              3.10.8>=3.3.0
opencv-python           4.13.0.90>=4.6.0
pillow                  12.1.0>=7.1.2
pyyaml                  6.0.3>=5.3.1
requests                2.32.5>=2.23.0
scipy                   1.17.0>=1.4.1
torch                   2.7.1+cu118>=1.8.0
torch                   2.7.1+cu118!=2.4.0,>=1.8.0; sys_platform == "win32"
torchvision             0.22.1+cu118>=0.9.0
psutil                  7.2.1>=5.8.0
polars                  1.37.1>=0.20.0
ultralytics-thop        2.0.18>=2.0.18
```
