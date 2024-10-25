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

learning_rate = 5e-4 # terlalu tinggi gak konsisten terlalu kecil lambat (gak ada aturan resmi berapa nya)
minibatch_size = 100
discount_factor = 0.99
replay_buffer_size = int(1e5) # berapa banayak pengalaaman yang bisa disempan ai
interpolation_parameter = 1e-3 #parameter yang mengendalikan seberapa besar kontribusi model lokal terhadap pembaruan model target nilai ini(0.001)

class ReplayMemory(object):
    

    def __init__(self, capacity):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.capacity = capacity
        self.memory = [] # daftar experience / pengalaman AI

    # fungsi yang akan menambakan data ke dalam memory
    def push(self, event): # event = pengalaman yang akan ditambah ke memory
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    #fungsi yang membuat sample untuk di train
    def sample(self,batch_size):
        experiences= random.sample(self.memory,k=batch_size)
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(self.device) # stak dari kondisi saat ini
        action= torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(self.device) # stak dari aksi nya
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self.device) # array / stak hadiah
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(self.device) # stak dari kondisi  selanjutnya
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device) # stak dari done
        return states,next_states,action,rewards,dones

class Agent():
    
    
    def __init__(self, state_size, action_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size

        # Inisialisasi dua jaringan saraf tiruan (Q-networks)
        self.local_qnetwork = Network(state_size, action_size).to(self.device) # Mewakili perkiraan nilai (Q-values) saat ini yang diperoleh oleh agen.
        self.target_qnetwork = Network(state_size, action_size).to(self.device) # nilai yang igin di deketi oleh local dan ini diperbaharui secara periodik sehinga lebih stabil
        #nilai target yang lebih stabil membantu mencegah nilai Q-values dari "melompat-lompat" secara berlebihan selama pelatihan.

        # Inisialisasi optimizer untuk mengoptimalkan parameter jaringan saraf lokal
        self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr=learning_rate)
        # Inisialisasi replay memory untuk menyimpan pengalaman
        self.memory = ReplayMemory(replay_buffer_size)
        # Inisialisasi langkah waktu untuk update target Q-network
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0:
            if len(self.memory.memory) > minibatch_size:
                experiences = self.memory.sample(100)
                self.learn(experiences, discount_factor)

    def act(self, state, epsilon = 0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.local_qnetwork.eval()
        with torch.no_grad():
            action_values = self.local_qnetwork(state)
        self.local_qnetwork.train()
        if random.random() > epsilon:
            #ini untuk exploitasi dimana aksi di tentukan dari probabilitas terbaik
            return np.argmax(action_values.cpu().data.numpy()) # mengambil data lalu di ubah ke numpy dan ambil nilai terbesar nya
        else:
            #ini untuk explorasi dimana aksi ditentukan benar benar random tanpa perhitungan
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, discount_factor):
        states, next_states, actions, rewards, dones = experiences
        next_q_targets = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1) # predisi nilai dari setiap action yang didapat untuk setiap sample
        q_targets = rewards + discount_factor * next_q_targets * (1 - dones)# nilai q target 
        q_expected = self.local_qnetwork(states).gather(1, actions) #mengembalikan tensor nilai dari setiap aksi dari setiap sample
        loss = F.mse_loss(q_expected, q_targets)# hitung loss, ini memberikan indikasi seberapa besar perbedaan antara nilai-nilai yang diprediksi oleh model
        self.optimizer.zero_grad() # mengatur ulang gradien sebelumnya jadi 0, agar gradien yang dihitung pada iterasi saat ini tidak diakumulasi dengan gradien sebelumnya.
        loss.backward() # menghitung gradien loss terhadap semua parameter model
        self.optimizer.step() # Optimizer menggunakan gradien yang dihitung selama backpropagation untuk memperbarui nilai parameter model
        self.soft_update(self.local_qnetwork, self.target_qnetwork, interpolation_parameter)   #digunakan untuk memperbarui parameter model target 


    #parameter-model target diubah secara perlahan menuju parameter-model lokal. Semakin kecil nilai interpolation_parameter,
    # semakin lambat pembaruan model target menuju model lokal.
    # Nilai yang umumnya digunakan untuk interpolation_parameter adalah sekitar 0.001 hingga 0.01
    def soft_update(self, local_model, target_model, interpolation_parameter):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            #interpolation_parameter * local_param.data: Kontribusi dari parameter lokal.
            #(1.0 - interpolation_parameter) * target_param.data: Kontribusi dari parameter target saat ini.
            target_param.data.copy_(interpolation_parameter * local_param.data + (1.0 - interpolation_parameter) * target_param.data)

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

## trainin AI
number_episodes = 2000
maxStep_perEpisode = 1000
epsilon_starting_value  = 1.0 #epsinon awal 1 = exploration
epsilon_ending_value  = 0.01 #end number =0.01= exploitasi
epsilon_decay_value  = 0.995 #number yang akan di kali epsilon unutk mengurangin nya secara bertahap
epsilon = epsilon_starting_value
scores_on_100_episodes = deque(maxlen = 100) #bikin queue yang berisi record 100 eposode terahir 

for episode in range(1, number_episodes + 1):
  state, _ = env.reset()
  score = 0
  for t in range(maxStep_perEpisode):
    action = agent.act(state, epsilon)
    next_state, reward, done, _, _ = env.step(action)
    agent.step(state, action, reward, next_state, done)
    state = next_state
    score += reward
    if done:
      break
  scores_on_100_episodes.append(score)
  epsilon = max(epsilon_ending_value, epsilon_decay_value * epsilon)
  print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)), end = "")
  if episode % 100 == 0:
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)))
  if np.array(scores_on_100_episodes).mean() >= 300.0:
    print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode - 100, np.mean(scores_on_100_episodes)))
    torch.save(agent.local_qnetwork.state_dict(), 'checkpoint1.pth')
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
    imageio.mimsave('video.mp4', frames, fps=30)

show_video_of_model(agent, 'LunarLander-v2')