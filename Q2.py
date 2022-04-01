import matplotlib.pyplot as plt
import numpy as np
import matplotlib.markers as markers
import copy


def add_dir(cell,direction):
    return (cell[0]+direction[0],cell[1]+direction[1])


class MDP_grid:
    
    def __init__(self,discount,eps,epi_length,alpha):
        
        
        ## Q is the required Q mapping.
        self.Q = {}
        self.grids = set()
        self.rewards_episode= []
        
        self.discount = discount
        self.eps = eps
        self.epi_length = epi_length
        self.alpha = alpha
        
        for i in range(0,50):
            self.grids.add((i,0))
            self.grids.add((i,24))

        for i in range(0,24):
            self.grids.add((0,i))
            self.grids.add((49,i))
            
        for i in range(0,12):
            self.grids.add((25,i))
            self.grids.add((26,i))
            
        for i in range(13,25):
            self.grids.add((25,i))
            self.grids.add((26,i))

        self.moves = ["UP","DOWN","LEFT","RIGHT"]
        
        self.move_shift = {}
        self.move_shift["UP"] = (0,1)
        self.move_shift["DOWN"] = (0,-1)
        self.move_shift["LEFT"] = (-1,0)
        self.move_shift["RIGHT"] = (1,0)
        
        self.goal = (48,12)
        
        for i in range(0,50):
            for j in range(0,25):
                state = (i,j)
                for act in self.moves:
                    self.Q[(state,act)] = 0
    
    
    def actual_action(self,action):
        
        prob_move = []
        for i in range(0,4):
            if self.moves[i] == action :
                prob_move.append(0.8)
            else :
                prob_move.append((0.2)/3)
        prob_move = np.array(prob_move)
        return np.random.choice(self.moves,p=prob_move)
    
    
    def chosen_action_greedily(self,state):
        
        all_q_values = [self.Q[(state, a)] for a in self.moves]
        max_q_value = max(all_q_values)
        pos_actions = [a for a in self.moves if self.Q[(state, a)] == max_q_value]
        best_action = np.random.choice(pos_actions, 1)[0]
        weights = [self.eps/len(self.moves) if a != best_action else 1 - self.eps + self.eps/len(self.moves) for a in self.moves]
        return np.random.choice(self.moves, 1, p = weights)[0]
            
    
    def create_episode(self):
        
        x, y = None,None
        while(True):
            x = np.random.randint(0,50)
            y = np.random.randint(0,25)
            if (x,y) not in self.grids and (x,y)!= self.goal:
                break
            
        state_history = []
        reward_history = []
        state_history.append((x,y))
        
        for i in range(0,self.epi_length):
            
            state = (x,y)
            if state == self.goal :
                break 
            
            action = self.chosen_action_greedily(state)
            actual = self.actual_action(action)
            
            new_state = add_dir(state,self.move_shift[actual])
            if new_state == self.goal:
                reward_history.append(100)
            
            elif new_state in self.grids:
                new_state = (x,y)
                reward_history.append(-1)
            
            else:
                reward_history.append(0)
            
            state_history.append(new_state)
            ### Q update step    
            val = max([self.Q[(new_state,act)] for act in self.moves])
            val = self.discount*val + reward_history[-1]
            val -= self.Q[(state,actual)] 
            
            self.Q[(state,actual)] = self.Q[(state,actual)] + self.alpha * val
            (x,y) = new_state
        
        self.rewards_episode.append(np.sum(np.array(reward_history)))
            
    def create_value_opt(self):
        
        self.values_obtained = np.zeros((50,25))
        self.optimal_policy = [[None for j in range(25)] for i in range(50)]
        
        for i in range(0,50):
            for j in range(0,25):
                
                if (i,j) in self.grids:
                    continue
                
                state = (i,j)
                val = -999999999.999
                pref_act = None
                
                for act in self.moves:
                    if self.Q[(state,act)] > val:
                        val = self.Q[(state,act)]
                        pref_act = act
                        
                self.values_obtained[i][j]= val
                self.optimal_policy[i][j] = pref_act
                

def arrow_coordinates(i, j, action):
    if action == 'UP':
        return i, j + 0.2, 0, 0.6
    elif action == 'DOWN':
        return i, j - 0.2, 0, -0.6
    elif action == 'LEFT':
        return i - 0.2, j, -0.6, 0
    elif action == 'RIGHT':
        return i + 0.2, j, 0.6, 0
 
    
grid1 = MDP_grid(0.99,0.05,1000,0.25)
for i in range(0,4000):
    grid1.create_episode()
grid1.create_value_opt()
                    
all_arrows = []
for i in range(50):
    for j in range(25):
        if grid1.optimal_policy[i][j] != None:
            x, y, dx, dy = arrow_coordinates(i, j, grid1.optimal_policy[i][j])
            all_arrows.append((x, y, dx, dy))
                
fig = plt.figure(figsize=(16, 8))
ax = fig.gca()
ax.set_xticks(np.arange(0, 51, 1))
ax.set_yticks(np.arange(0, 26, 1))
ax.set_xlim([-0.5, 49.5])
ax.set_ylim([-0.5, 24.5])
marker = markers.MarkerStyle(marker='s')
P = np.arange(25)
Q = np.arange(50)
all_points = np.dstack(np.meshgrid(P, Q)).reshape(-1, 2)
X = grid1.values_obtained.reshape(-1, 1)
x  = (X - np.min(X))/ (np.max(X)- np.min(X)) * 255
all_walls = list(grid1.grids)
wall_color = ['r']*len(all_walls)


scat1 = plt.scatter(all_points[:, 1], all_points[:, 0], s=200, c=X[:, 0], cmap='Greys', edgecolors='k', marker=marker)
scat2 = plt.scatter([x[0] for x in all_walls], [x[1] for x in all_walls], s=200, c=wall_color, edgecolors='k', marker=marker)
for i in range(len(all_arrows)):
    plt.arrow(all_arrows[i][0], all_arrows[i][1], all_arrows[i][2], all_arrows[i][3], length_includes_head=True, head_width=0.15, edgecolor='b', facecolor='y')

# plt.grid()
plt.savefig('b_0.05.png', bbox_inches='tight')
plt.show()

##Grid 2 and grid 3

grid2 = MDP_grid(0.99,0.005,1000,0.25)
for i in range(0,4000):
    grid2.create_episode()
grid2.create_value_opt()
                    
all_arrows_2 = []
for i in range(50):
    for j in range(25):
        if grid2.optimal_policy[i][j] != None:
            x, y, dx, dy = arrow_coordinates(i, j, grid2.optimal_policy[i][j])
            all_arrows_2.append((x, y, dx, dy))
                
fig = plt.figure(figsize=(16, 8))
ax = fig.gca()
ax.set_xticks(np.arange(0, 51, 1))
ax.set_yticks(np.arange(0, 26, 1))
ax.set_xlim([-0.5, 49.5])
ax.set_ylim([-0.5, 24.5])
marker = markers.MarkerStyle(marker='s')
P = np.arange(25)
Q = np.arange(50)
all_points = np.dstack(np.meshgrid(P, Q)).reshape(-1, 2)
X = grid2.values_obtained.reshape(-1, 1)
x  = (X - np.min(X))/ (np.max(X)- np.min(X)) * 255
all_walls = list(grid2.grids)
wall_color = ['r']*len(all_walls)


scat1 = plt.scatter(all_points[:, 1], all_points[:, 0], s=200, c=X[:, 0], cmap='Greys', edgecolors='k', marker=marker)
scat2 = plt.scatter([x[0] for x in all_walls], [x[1] for x in all_walls], s=200, c=wall_color, edgecolors='k', marker=marker)
for i in range(len(all_arrows)):
    plt.arrow(all_arrows_2[i][0], all_arrows_2[i][1], all_arrows_2[i][2], all_arrows_2[i][3], length_includes_head=True, head_width=0.15, edgecolor='b', facecolor='y')

# plt.grid()
plt.savefig('b_0.005.png', bbox_inches='tight')
plt.show()

grid3 = MDP_grid(0.99,0.5,1000,0.25)
for i in range(0,4000):
    grid3.create_episode()
grid3.create_value_opt()
                    
all_arrows_3 = []
for i in range(50):
    for j in range(25):
        if grid3.optimal_policy[i][j] != None:
            x, y, dx, dy = arrow_coordinates(i, j, grid3.optimal_policy[i][j])
            all_arrows_3.append((x, y, dx, dy))
                
fig = plt.figure(figsize=(16, 8))
ax = fig.gca()
ax.set_xticks(np.arange(0, 51, 1))
ax.set_yticks(np.arange(0, 26, 1))
ax.set_xlim([-0.5, 49.5])
ax.set_ylim([-0.5, 24.5])
marker = markers.MarkerStyle(marker='s')
P = np.arange(25)
Q = np.arange(50)
all_points = np.dstack(np.meshgrid(P, Q)).reshape(-1, 2)
X = grid3.values_obtained.reshape(-1, 1)
x  = (X - np.min(X))/ (np.max(X)- np.min(X)) * 255
all_walls = list(grid3.grids)
wall_color = ['r']*len(all_walls)


scat1 = plt.scatter(all_points[:, 1], all_points[:, 0], s=200, c=X[:, 0], cmap='Greys', edgecolors='k', marker=marker)
scat2 = plt.scatter([x[0] for x in all_walls], [x[1] for x in all_walls], s=200, c=wall_color, edgecolors='k', marker=marker)
for i in range(len(all_arrows)):
    plt.arrow(all_arrows_3[i][0], all_arrows_3[i][1], all_arrows_3[i][2], all_arrows_3[i][3], length_includes_head=True, head_width=0.15, edgecolor='b', facecolor='y')

# plt.grid()
plt.savefig('b_0.5.png', bbox_inches='tight')
plt.show()


reward_per_epoch_explore = np.array(grid3.rewards_episode)
for i in range(1,4000):
    reward_per_epoch_explore[i] += reward_per_epoch_explore[i-1]
reward_per_epoch_non_explore = np.array(grid1.rewards_episode)
for i in range(1,4000):
    reward_per_epoch_non_explore[i] += reward_per_epoch_non_explore[i-1]


store = np.array([i for i in range(1,4001)])
reward_per_epoch_explore = reward_per_epoch_explore/store
reward_per_epoch_non_explore = reward_per_epoch_non_explore/store
x_axis = np.array([i for i in range(1,4001)])

plt.xlabel("Iterations")
plt.ylabel("Reward per epoch")
plt.plot(x_axis,reward_per_epoch_explore,label="eps=0.5",color="red")
plt.plot(x_axis,reward_per_epoch_non_explore,label="eps=0.05",color="blue")
plt.legend()
plt.savefig("c.png")
plt.show()

      
