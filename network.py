import torch as T 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
import numpy as np 
from sklearn import preprocessing


class DeepQNetwork(nn.Module):
  def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
    super(DeepQNetwork, self).__init__()
    self.input_dims = input_dims 
    self.fc1_dims = fc1_dims 
    self.fc2_dims = fc2_dims 
    self.actions = n_actions 
    self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
    self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
    self.fc3 = nn.Linear(self.fc2_dims, self.fc2_dims)
    self.fc4 = nn.Linear(self.fc2_dims, self.fc2_dims)
    self.fc5 = nn.Linear(self.fc2_dims, n_actions)
    self.optimizer = optim.Adam(self.parameters(), lr=lr)
    self.loss = nn.MSELoss()
    self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    self.to(self.device)
  
  def forward(self, state):
    x = F.relu(self.fc1(state))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = F.relu(self.fc4(x))

    actions = self.fc5(x)
    return actions 


class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size=100000, 
    eps_end=0.01, eps_dec=3e-4, training_mode=True):
      self.gamma = gamma 
      self.epsilon = epsilon
      self.lr = lr  
      self.eps_min = eps_end
      self.eps_dec = eps_dec  
      self.action_space =  [i for i in range(n_actions)]
      self.mem_size = max_mem_size 
      self.batch_size = batch_size 
      self.mem_counter = 0 
      self.q_eval = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims, 
      fc1_dims=128, fc2_dims=128) 

      if not training_mode:
        print('Carregando modelo')
        self.q_eval.load_state_dict(T.load("./checkpoint/checkpoint_6000_treino2.pth"))
      
      self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
      self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)

      self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
      self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
      self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)


    def store_transition(self, state, action, reward, state_, done):
      index = self.mem_counter % self.mem_size 
      self.state_memory[index] = state 
      self.new_state_memory[index] = state_
      self.reward_memory[index] = reward 
      self.action_memory[index] = action 
      self.terminal_memory[index] = done 

      self.mem_counter += 1


    def choose_action(self, observation, valid_actions):
      if np.random.random() > self.epsilon:
        print('Ação Deep Q Network')
        state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
        # print('NETWORK INPUT {}'.format(state))
        actions = self.q_eval.forward(state)

        # transform q-value for illegal actions into zero
        actions = self.transform_valid_out(actions[0], valid_actions)
        action = T.argmax(actions).item()
      else:
        print('Ação por exploração aleatória')
        action = np.random.choice(valid_actions)

      return action 


    def transform_valid_out(self, output_values, valid_actions):
      # print('VALID ACTION: {}'.format(valid_actions))
      for index in range(len(output_values)):
        if index not in valid_actions:
          output_values[index] = 0
        # print('OUT {}:{}'.format(index, output_values[index]))

      return output_values


    def learn(self):
      # O número de tuplas armazenadas precisa ser maior que o tamanho do batch para o agente começar a aprender
      if self.mem_counter < self.batch_size:
        return 

      self.q_eval.optimizer.zero_grad()

      # Pega o tamanho atual da memória
      max_mem = min(self.mem_counter, self.mem_size)

      # Seleciona 64 índices aleatórios da memória de replay
      batch = np.random.choice(max_mem, self.batch_size, replace=False)
      
      # Gera os índices para o lote 
      batch_index = np.arange(self.batch_size, dtype=np.int32)

      # seleciona as 64 amostras da memória de replay
      state_batch = T.tensor(self.state_memory[batch]).to(self.q_eval.device)
      new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.q_eval.device)
      reward_batch = T.tensor(self.reward_memory[batch]).to(self.q_eval.device)
      terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.q_eval.device)

      action_batch = self.action_memory[batch]

      # propaga os estados na rede
      q_eval = self.q_eval.forward(state_batch)[batch_index, action_batch]
      q_next = self.q_eval.forward(new_state_batch)
      q_next[terminal_batch] = 0.0

      # calcula os q-targets para a rede
      q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

      # calcula o erro,  (valor Q fornecido pela rede - valor q calculado pela fórmula)
      loss = self.q_eval.loss(q_target, q_eval).to(self.q_eval.device)
      loss.backward()
      self.q_eval.optimizer.step()

      print('LOSS: {}'.format(loss.item()))
      
      # epsilon é decrementado, ao longo do tempo a rede passa a priorizar uma política gulosa
      self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    
