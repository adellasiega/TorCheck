import torch
from torcheck import stl

print("CUDA availabe: " + str(torch.cuda.is_available()))
device = torch.device("cuda")

signal1 = torch.randn(1, 2, 100)
print(signal1.shape)

# phi(x) = (x1>=1) and (globally[0,4] x2<=2)
n0 = stl.Atom(var_index=0, threshold=1, lte=False)  # lte = False is >=
n1 = stl.Atom(var_index=1, threshold=2, lte=True)   # lte = True is <=
ng = stl.Globally(n1, left_time_bound=0, right_time_bound=4)
phi = stl.And(n0, ng)

print(phi)

# signal dimension [n_samples, n_vars, n_timesteps]
z0 = phi.boolean(signal1)
print(z0)

z1 = phi.quantitative(signal1)
print(z1)

# psi(x) = (x1>=1) until [0,4] (x2<=2)
n0 = stl.Atom(var_index=0, threshold=1, lte=False)  # lte = False is >
n1 = stl.Atom(var_index=1, threshold=2, lte=True)   # lte = True is <
psi = stl.Until(n0, n1, left_time_bound=0, right_time_bound=4)
print(psi)
print(psi.boolean(signal1))
print(psi.quantitative(signal1, normalize=False))
