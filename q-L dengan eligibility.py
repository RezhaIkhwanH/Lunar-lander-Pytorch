
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from collections import deque, namedtuple
import numpy as np
from collections import namedtuple, deque
# Defining one Step
Step = namedtuple('Step', ['state', 'action', 'reward', 'done'])
# Making the AI progress on several (n_step) steps
class NStepProgress:
    
    def __init__(self, env, ai, n_step):
        self.ai = ai
        self.rewards = []
        self.env = env
        self.n_step = n_step
    
    def __iter__(self):
        state,_ = self.env.reset()
        history = deque()
        reward = 0.0
        while True:
            action = self.ai.act(state)
            next_state, r, is_done, _, _ = env.step(action)
            reward += r
            history.append(Step(state = state, action = action, reward = r, done = is_done))
            while len(history) > self.n_step + 1:
                history.popleft()
            if len(history) == self.n_step + 1:
                yield tuple(history)
            state = next_state
            if is_done:
                if len(history) > self.n_step + 1:
                    history.popleft()
                while len(history) >= 1:
                    yield tuple(history)
                    history.popleft()
                self.rewards.append(reward)
                reward = 0.0
                state,_ = self.env.reset()
                history.clear()
    
    def rewards_steps(self):
        rewards_steps = self.rewards
        self.rewards = []
        return rewards_steps

# Implementing Experience Replay

class ReplayMemory:
    
    def __init__(self, n_steps, capacity = 10000):
        self.capacity = capacity
        self.n_steps = n_steps
        self.n_steps_iter = iter(n_steps)
        self.buffer = deque()

    def sample_batch(self, batch_size): # creates an iterator that returns random batches
        ofs = 0
        vals = list(self.buffer)
        np.random.shuffle(vals)
        while (ofs+1)*batch_size <= len(self.buffer):
            yield vals[ofs*batch_size:(ofs+1)*batch_size]
            ofs += 1

    def run_steps(self, samples):
        while samples > 0:
            entry = next(self.n_steps_iter) # 10 consecutive steps
            self.buffer.append(entry) # we put 200 for the current episode
            samples -= 1
        while len(self.buffer) > self.capacity: # we accumulate no more than the capacity (10000)
            self.buffer.popleft()


class Network(nn.Module):

    def __init__(self,state_size,action_size,seed=42):
        super(Network,self).__init__()
        self.seed=torch.manual_seed(seed)
        self.fc1=nn.Linear(state_size,64) #hiden layer 1
        self.fc2=nn.Linear(64,64) #hiden layer 2
        self.fc3=nn.Linear(64,action_size) #output layer

    def forward(self,state):
        # state= input
        x= self.fc1(state)# input masuk ke fc1 / layer 1
        x= F.relu(x) #hasil nya di masukan ke fungsi rellu
        x=self.fc2(x) # lalu masuk ke layer selanjutnya
        x=F.relu(x)
        return self.fc3(x) # hasil nya akan masuk ke output layer dan di return

#setup env
import gymnasium as gym
env = gym.make('LunarLander-v2')
state_shape=env.observation_space.shape # ?
state_size=env.observation_space.shape[0] #berapa input nya
number_actions=env.action_space.n # berapa action / ouput
print('State shape: ', state_shape)
print('State size: ', state_size)
print('Number of actions: ', number_actions)

learning_rate = 5e-4 # terlalu tinggi gak konsisten terlalu kecil lambat
minibatch_size = 100
discount_factor = 0.99
replay_buffer_size = int(1e5) # berapa banayak pengalaaman yang bisa disimpan ai
interpolation_parameter = 1e-3 #parameter yang mengendalikan seberapa besar kontribusi model lokal terhadap pembaruan model target nilai ini(0.001)


class Agent():
    
    
    def __init__(self, state_size, action_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size

        self.local_qnetwork = Network(state_size, action_size).to(self.device) 
        
        self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr=learning_rate)
        

    def act(self, state, epsilon = 0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.local_qnetwork.eval()
        with torch.no_grad():
            action_values = self.local_qnetwork(state)
        self.local_qnetwork.train()
        probs = F.softmax(action_values * 1.0, dim=1)
        # Menggunakan distribusi multinomial untuk memilih aksi
        actions = probs.multinomial(num_samples=1)
        return actions.item()

    

    

    def load_model(self, file_path):
        # Muat kembali model dari file checkpoint
        if os.path.isfile(file_path):
            
            self.local_qnetwork.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))
            # Inisialisasi optimizer
            self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr=learning_rate)
            # Setel ulang langkah waktu
            self.t_step = 0
            print("Model berhasil dimuat dari {}".format(file_path))
        else:
            print("File {} tidak ditemukan. Model tidak dapat dimuat.".format(file_path))


agent=Agent(state_size,number_actions)
# agent.load_model("checkpoint.pth")

# Implementing Eligibility Trace
def eligibility_trace(batch):
    gamma = 0.99
    inputs = []
    targets = []
    for series in batch:
        input = torch.from_numpy(np.array([series[0].state, series[-1].state], dtype = np.float32)).to(agent.device)
        
        output = agent.local_qnetwork(input)
        cumul_reward = 0.0 if series[-1].done else output[1].data.max()
        for step in reversed(series[:-1]):
            cumul_reward = step.reward + gamma * cumul_reward
        state = series[0].state
        target = output[0].data
        target[series[0].action] = cumul_reward
        inputs.append(state)
        targets.append(target)
    return torch.from_numpy(np.array(inputs, dtype = np.float32)).to(agent.device), torch.stack(targets).to(agent.device)




## trainin AI
number_episodes = 2000
maxStep_perEpisode = 1000
scores_on_100_episodes = deque(maxlen = 100) #bikin queue yang berisi record 100 eposode terahir 


n_steps = NStepProgress(env = env, ai = agent, n_step = 10)
memory = ReplayMemory(n_steps = n_steps, capacity = 10000)

for episode in range(1, number_episodes + 1):
  memory.run_steps(200)
  for batch in memory.sample_batch(128):
      
      inputs, targets = eligibility_trace(batch)
      
      predictions = agent.local_qnetwork(inputs)
      loss = F.mse_loss(predictions, targets)
      agent.optimizer.zero_grad()
      loss.backward()
      agent.optimizer.step()
      
  scores_on_100_episodes.extend(list(n_steps.rewards))
  
  print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)), end = "")
  if episode % 100 == 0:
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)))
  if np.array(scores_on_100_episodes).mean() >= -200.0:
    print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode - 100, np.mean(scores_on_100_episodes)))
    torch.save(agent.local_qnetwork.state_dict(), 'checkpoint_eligibility.pth')
    break



import glob
import io
import base64
import imageio
from gym.wrappers.monitoring.video_recorder import VideoRecorder

def show_video_of_model(agent, env_name):
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    frames = []
    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action)
    env.close()
    imageio.mimsave('video2.mp4', frames, fps=30)

show_video_of_model(agent, 'LunarLander-v2')