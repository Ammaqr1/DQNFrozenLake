import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn 
import torch.nn.functional as F



## model 
class DQN(nn.Module):
    def __init__(self,in_states,h1_nodes,out_actions):
        super().__init__()
        self.fc1 = nn.Linear(in_states,h1_nodes)
        self.out = nn.Linear(in_states,out_actions)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x
    
# define memory for experience replay
class replaymemory():
    def __init__(self,maxlen):
        self.memory = deque([],maxlen= maxlen)
        
    def append(self,transition):
        self.memory.append(transition)
    
    def sample(self,sample_size):
        return random.sample(self.memory,sample_size)
    
    def __len__(self):
        return len(self.memory)

## frozen lake deep q learning
class frozenlakedql():
    #hyperparameter
    learning_rate_a = 0.001
    discount_factor_g = 0.9
    network_sync_rate = 10    
    replay_memory_size = 1000  
    mini_batch_size = 32        
        
    ##Neural network
    loss_fn = nn.MSELoss()
    optimizer = None  

    ACTIONS = ['L','D','R','U'] 


    def train(self,episodes,render = False,is_slippery = False):
        #create Frozen instance
        env = gym.make('FrozenLake-v1',map_name = '4x4',is_slippery=is_slippery,render_mode = 'human' if render else None)
        num_states = env.observation_space.n
        num_actions = env.action_space.n
            
        epsilon = 1
        memory = replaymemory(self.replay_memory_size)
            
        policy_dqn = DQN(in_states=num_states,h1_nodes=num_states,out_actions=num_actions)
        target_dqn = DQN(in_states=num_states,h1_nodes=num_states,out_actions=num_actions)


        target_dqn.load_state_dict(policy_dqn.state_dict())
            
        print('Policy (random,before training): ')
        self.print_dqn(policy_dqn)
        
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(),lr = self.learning_rate_a)
        
        rewards_per_episode = np.zeros(episodes)
        
        epsilon_history = []
        
        step_count = 0
            

        for i in range(episodes):
            if i % 10 == 0:
                print(i)
            state = env.reset()[0] 
            terminated = False
            truncated = False 
            # AGent navigates map until it falls into hole/ reaches gaal (terminated),or has taken 200 actoins (truncated).
            while(not terminated and not truncated):
                # select action bsed on epsilon-greedy
                if random.random() < epsilon:
                    #select random action
                    action = env.action_space.sample()#actions:0=left,1=down,2=right,3=up
                    
                else:
                    with torch.no_grad():
                        action = policy_dqn(self.state_to_dqn_input(state,num_states)).argmax().item()
                        
                # execute action
                new_state,reward,terminated,truncated,_ = env.step(action)
                
                #save experience into memory
                memory.append((state,action,new_state,reward,terminated))
                
                state = new_state
                step_count += 1
             
            if reward == 1:
                rewards_per_episode[i] = 1
                
            if len(memory) > self.mini_batch_size and np.sum(rewards_per_episode)>0:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch,policy_dqn,target_dqn)
                
                epsilon = max(epsilon - 1/ episodes,0)
                epsilon_history.append(epsilon)
                
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    
        # Close environment
        env.close()
        
        # save policy 
        torch.save(policy_dqn.state_dict(),'frozen_lake_dqn.pt')
        
        # create new graph
        plt.figure(1)
        
        ## plot average rewards (y-axis) vs episodes (x-axis)
        sum_rewards = np.zeros(episodes)
        for x in range(episodes):
            sum_rewards[x] = np.sum(rewards_per_episode[max(0,x-100):(x+1)])
            
        plt.subplot(121) # plot on a 1 row x 2 col grid , at cell 1
        plt.plot(sum_rewards)
        
        # plot epsilon decay (y-axis) vs episodes (x-axis)
        plt.subplot(122)
        plt.plot(epsilon_history)

        # Save plots
        plt.savefig('frozen_lake_dqn.png')
       
        

    
    def optimize(self,mini_batch,policy_dqn,target_dqn):
        
        # get no of input nodes
        num_states = policy_dqn.fc1.in_features
            
        current_q_list = []
        target_q_list = []
            
        for state,action ,new_state,reward,terminated in mini_batch:
                
            if terminated:
                target = torch.FloatTensor([reward])
            else:
                # calculate target q value
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + self.discount_factor_g * target_dqn(self.state_to_dqn_input(new_state,num_states)).max()
                        )
                        
            # get the current set of q values
            current_q = policy_dqn(self.state_to_dqn_input(state,num_states))
            current_q_list.append(current_q)
                        

            ## get the target set of q values
            target_q = target_dqn(self.state_to_dqn_input(state,num_states))
            # Adjust the specific action to the target thtat was just calculated 
            target_q[action] = target
            target_q_list.append(target_q)
                        

                # compute loss for the whole minibatch
        loss = self.loss_fn(torch.stack(current_q_list),torch.stack(target_q_list)) 
                    
        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
                        
        
    def state_to_dqn_input(self,state:int,num_states:int)-> torch.Tensor:
        input_tensor = torch.zeros(num_states)
        input_tensor[state] = 1
        return input_tensor
        
    # Run the Frozenlake environment with the learned policy
        
    def test(self,episodes,is_slippery= False):
        # create frozen lake instance
        env = gym.make('FrozenLake-v1',map_name = '4x4',is_slippery= is_slippery,render_mode= 'human')
        num_states = env.observation_space.n
        num_actions = env.action_space.n
            
        # Load learned policy
        policy_dqn = DQN(in_states=num_states,h1_nodes= num_states,out_actions = num_actions)
        policy_dqn.load_state_dict(torch.load('frozen_lake_dqn.pt'))
        policy_dqn.eval() # switch model to evaluation mode
            
        print('policy (trained):')
        self.print_dqn(policy_dqn)
            
        for i in range(episodes):
            if i % 10 == 0:
                print(i)
            state = env.reset()[0]
            terminated = False
            truncated = False

            # Agent navigates map until it falls into a hole (terminated), reaches goal (terminated), or has taken 200 actions (truncated).

            while(not terminated and not truncated):
                
                with torch.no_grad():
                    # select best acion
                    action = policy_dqn(self.state_to_dqn_input(state,num_states)).argmax().item()
                    
                # execute action
                state,reward,terminated,truncated,_ = env.step(action)
                
        env.close()
        
    # Print DQN: state, best action, q values
    def print_dqn(self, dqn):
        # Get number of input nodes
        num_states = dqn.fc1.in_features

        # Loop each state and print policy to console
        for s in range(num_states):
            #  Format q values for printing
            q_values = ''
            for q in dqn(self.state_to_dqn_input(s, num_states)).tolist():
                q_values += "{:+.2f}".format(q)+' '  # Concatenate q values, format to 2 decimals
            q_values=q_values.rstrip()              # Remove space at the end

            # Map the best action to L D R U
            best_action = self.ACTIONS[dqn(self.state_to_dqn_input(s, num_states)).argmax()]

            # Print policy in the format of: state, action, q values
            # The printed layout matches the FrozenLake map.
            print(f'{s:02},{best_action},[{q_values}]', end=' ')         
            if (s+1)%4==0:
                print() # Print a newline every 4 states

if __name__ == '__main__':

    frozen_lake = frozenlakedql()
    is_slippery = False
    # frozen_lake.train(1000, is_slippery=is_slippery)
    frozen_lake.test(10, is_slippery=is_slippery)  
            

                
                
                

                
        
 
