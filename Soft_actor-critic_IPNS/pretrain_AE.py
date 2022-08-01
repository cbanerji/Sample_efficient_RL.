import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from numpy import random
import gym
import numpy as np

#-------------------------Generate data-------------
env = gym.make('Hopper-v2')
observation = env.reset()
tstep = 0
t = 0
obs_col = []
act_col = []

while tstep <10000:
    t += 1
    tstep +=1
    action = env.action_space.sample()
    act_col.append(action)
    observation, reward, done, info = env.step(action)
    obs_col.append(observation)
    print(tstep)
    if done:
        print("Episode finished after {} timesteps".format(t))
        observation = env.reset()
        t = 0
exp_dat = np.asarray(obs_col)
np.savetxt("s_Hopper.csv", exp_dat, delimiter=",")
vec_shape = exp_dat.shape

#--------------------Define dataset-------------------------
class StateDataset(Dataset):

    def __init__(self):
        # data loading
        #state = np.loadtxt('exp_dat_Reacher.csv', delimiter = ",", dtype= np.float32)
        state = np.loadtxt('s_Hopper.csv', delimiter = ",", dtype= np.float32)
        self.st = torch.from_numpy(state)
        self.n_samples = exp_dat.shape[0]

    def __getitem__(self, index):
        return self.st[index]

    def __len__(self):
        return self.n_samples

dataset = StateDataset()

# Dataloader loads data for training
loader = DataLoader(dataset = dataset,batch_size = 128, shuffle = True)
# Define Autoencoder architecture
class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.vec_shape = vec_shape

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.vec_shape[1], 64),
            torch.nn.ELU(),
            torch.nn.Linear(64, 16),
            torch.nn.ELU(),
            torch.nn.Linear(16,3),
            torch.nn.ELU(),
            torch.nn.Sigmoid(),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(3,16),
            torch.nn.ELU(),
            torch.nn.Linear(16,64),
            torch.nn.ELU(),
            torch.nn.Linear(64, self.vec_shape[1]),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model = AE() # Model Initialization
print(model)

loss_function = torch.nn.MSELoss()  # Validation using MSE Loss function
optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3,weight_decay = 1e-5) # Using an Adam Optimizer

#------------Training loop-------------------------------
epochs = 300
outputs = []
losses = []
for epoch in range(epochs):
    for i,(a) in enumerate(loader):
      reconstructed = model(a) # Output of Autoencoder

      # Calculating the loss function
      loss = loss_function(reconstructed,a)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # Storing the losses in a list for plotting
      losses.append(loss)
    print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
    outputs.append((epochs, a, reconstructed))

# Save Model
torch.save(model.state_dict(), "/home/Hopper_pretrain10k.pth")
# Defining the Plot Style
plt.xlabel('Iterations')
plt.ylabel('Loss')

# Plotting the last 100 values
plt.plot(losses[-100:])
#np.savetxt("loss_bneck4.csv", losses, delimiter=",")
plt.show()
