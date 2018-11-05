## DQN agent description

### Learning Algorithm
The objective of this project is to collect as many yellow bananas as possible, avoiding blue bananas. It's an eposidic problem where we want to maximize the expected reward.  The criteria for solving is considered to be an average score of +13 over 100 consecutive episodes.
As a starting point I used a Deep Q-Network agent with Prioritized Experience Replay as proposed during the class in the Lunar Lender project. To further improve the algorithm I adopted the Double DQN approach (to compensate overestimation of Q values) and the Dueling approach (to better understand the connection between initial state, action and rewards).

#### Dueling DQN
```
class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)

        self.advt = nn.Linear(fc2_units, action_size)
        self.value = nn.Linear(fc2_units, fc2_units)
        
        self.fc3 = nn.Linear((fc2_units + action_size), action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x1 = F.relu(self.advt(x))
        x2 = F.relu(self.value(x))
        x = torch.cat((x1, x2), 1)
        return self.fc3(x)
```

#### Double DQN
```
def learn(self, experiences, gamma, TAU):
        states, actions, rewards, next_states, dones = experiences

        # Calculate target value
        self.qnetwork_target.eval()
        with torch.no_grad():
            Q_local = self.qnetwork_local(next_states)
            Q_target = self.qnetwork_target(next_states)
            argmax_action = torch.max(Q_local, dim=1, keepdim=True)[1]
            Q_max = Q_target.gather(1, argmax_action)
            y = rewards + gamma * Q_max * (1 - dones)
        self.qnetwork_target.train()

        # Predict Q-value
        self.optimizer.zero_grad()
        Q = self.qnetwork_local(states)
        y_pred = Q.gather(1, actions)

        # TD-error
        loss = torch.sum((y - y_pred)**2)

        # Optimize
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)  
```

### Hyperparameters
```

```
### Neural network model architecture

### Plot of rewards per episode
![Plot](/images/DQNagent.PNG)

### Further improvements
As discussed during the class, to further improve the agent, I should implement a rainbow algorithm. The rainbow algorithm implements Double DNW, Dueling DQN and Prioritized experience replay as I did. In addition to that, it implements multi-step bootstrap targets, Distributional DQN and Noisy DQN.
We can see in the image how it performs in respect to the other approaches.
![Rainbow](/images/Rainbow.png)
