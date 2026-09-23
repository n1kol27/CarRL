import torch.optim as optim
from model import *


network = SpeedModel(7, 9)
optimizer = optim.Adam(network.parameters(), lr=0.001)

