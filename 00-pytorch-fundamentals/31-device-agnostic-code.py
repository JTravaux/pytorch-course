import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

# Create a tensor on CPU (default)
tensor_cpu = torch.tensor([1.0, 2.0, 3.0])
print(tensor_cpu, tensor_cpu.device)

# Move tensor to GPU if available
tensor_gpu = tensor_cpu.to(device)
print(tensor_gpu, tensor_gpu.device)

try:
    tensor_gpu.numpy() # Error: can't convert CUDA tensor to numpy (use Tensor.cpu() to copy the tensor to host memory first)
except Exception as e:
    print(e)

tensor_back_on_cpu = tensor_gpu.cpu();
print(tensor_back_on_cpu, tensor_back_on_cpu.device, tensor_back_on_cpu.numpy())
