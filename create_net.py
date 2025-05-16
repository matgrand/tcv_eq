import torch
import numpy as np
class TestNet(torch.nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.lin = torch.nn.Sequential(
            torch.nn.Linear(2, 512), torch.nn.ReLU(),
            torch.nn.Linear(512, 3),
        )
        with torch.no_grad(): # fill with 1 for reproducibility
            for m in self.modules():
                if isinstance(m, torch.nn.Linear):
                    m.weight.fill_(1.0)
                    m.bias.fill_(1.0)
    def forward(self, x):
        x = self.lin(x)
        return x
    
net = TestNet()
net = net.double()
net.eval()
x = torch.tensor([3.0,5.0], dtype=torch.float64).reshape(1,2)
y = net(x).detach()
np.set_printoptions(precision=4, suppress=True, sign='+')
x, y = x.numpy().reshape(-1), y.numpy().reshape(-1)
print(f'x -> {x}')
print(f'y -> {y}')

# convert to onnx
dummy_input = torch.randn(1, 2, dtype=torch.float64)
torch.onnx.export(net, dummy_input, 'net.onnx', export_params=True,
                  opset_version=12, do_constant_folding=True,
                  input_names=['x'], output_names=['y'])




## Evaluate network inference time 
from time import time
from tqdm import tqdm
N = 100_000
ts = np.zeros(N)
ys = np.zeros((N, 3))
for i in tqdm(range(N), leave=False):
    x = torch.randn(1, 2, dtype=torch.float64)
    start = time()
    y = net(x).detach()
    ts[i] = time() - start
    ys[i] = y.numpy().reshape(-1)
print(f'Testing inference time for {N} iterations...')
ts_us = ts * 1e6
print(f'Inference time -> {np.mean(ts_us):.1f} ± {np.std(ts_us):.1f} [μs] | max {np.max(ts_us):.1f} [μs]')

# plot histogram
import matplotlib.pyplot as plt
plt.hist(ts_us, bins=100)
plt.xlabel('Inference time [μs]')
plt.ylabel('Count')
plt.title('Inference time distribution')
plt.ylim(0, 100)
plt.grid()
#save figure
plt.savefig('test/python_inference_time.png')
