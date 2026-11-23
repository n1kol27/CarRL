import torch
import torch.optim as optim
from game.game import run_train, run
from model import SpeedModel, ReplayBuffer


network = SpeedModel(7, 7)
network.load_state_dict(torch.load("models/model.pth"))
optimizer = optim.Adam(network.parameters(), lr=0.001)
replay_buffer = ReplayBuffer(capacity=10000)


#run_train(network, optimizer, replay_buffer, car_count=15)
run()
torch.save(network.state_dict(), "models/model.pth")