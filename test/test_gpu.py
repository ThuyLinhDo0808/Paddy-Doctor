import torch
print(torch.cuda.is_available())  # should print: True
print(torch.cuda.get_device_name())  # should print: NVIDIA RTX 3060 Laptop GPU
