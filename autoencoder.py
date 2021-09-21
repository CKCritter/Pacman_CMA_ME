import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

import gym
env = gym.make('MsPacman-v0')
env.reset()

torch.set_default_dtype(torch.float64)

plt.figure(figsize=(9, 2))
plt.gray()
plt.axis("off")
_, sub = plt.subplots(2, 1);

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 10, (3, 3), padding=1),
            nn.ReLU(),
            #nn.Conv2d(100, 40, (9, 9), stride=2, padding=1),
            #nn.ReLU(),
            #nn.Conv2d(40, 20, (1, 1), padding=1)
        )
        
        self.decoder = nn.Sequential(
            #nn.ConvTranspose2d(20, 40, (1, 1), stride=1, padding=1, output_padding=1),
            #nn.ReLU(),
            #nn.ConvTranspose2d(40, 100, (9, 9), stride=1, padding=1, output_padding=1),
            #nn.ReLU(),
            nn.ConvTranspose2d(10, 3, (3, 3), padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model = Autoencoder()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=1e-3, 
                             weight_decay=1e-5)

# Point to training loop video
num_epochs = 1000
outputs = []
observation, reward, done, info = env.step(env.action_space.sample())
f = lambda x: x / 255
for epoch in range(num_epochs):
    #for i in range(0, 1000):
    observation, reward, done, info = env.step(env.action_space.sample())
    obs = observation.reshape(1, 3, 160, 210)
    
    obs = f(obs)
    obs = torch.from_numpy( obs.astype('double') )
    recon = model(obs)
    loss = criterion(recon, obs)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
    outputs.append((epoch, obs, recon))
    if done:
        env.reset()
        
for k in range(1001, num_epochs, 100):    
    imgs = outputs[k][1].detach().numpy()
    recon = outputs[k][2].detach().numpy()
    for i, item in enumerate(imgs):
        if i >= 2: break
        Z = item.reshape(210, 160, 3)
        sub[0].imshow(Z, interpolation='none', aspect='auto')

    for i, item in enumerate(recon):
        if i >= 2: break
        Z = item.reshape(210, 160, 3)
        sub[1].imshow(Z, interpolation='none', aspect='auto')
    plt.show()

import cv2
obs = f(cv2.imread("filename - Copy.png").reshape(1, 3, 160, 210))
sub[0].imshow(obs.reshape(210, 160, 3), interpolation='none', aspect='auto')
obs = torch.from_numpy( obs.astype('double') )
recon = model(obs)
sub[1].imshow(recon.detach().numpy().reshape(210, 160, 3), interpolation='none', aspect='auto')
plt.show()
print("hi")
