import matplotlib.pyplot as plt
import numpy as np
import matplotlib.markers as markers
import copy


def add_dir(cell,direction):
    return (cell[0]+direction[0],cell[1]+direction[1])


class MDP_grid:
    
    def __init__(self,discount,epi_length,iterations,theta):
        
        self.optimal_policy = [[None for j in range(25)] for i in range(50)]
        ## Q is the required Q mapping.
        self.Q = {}
        self.grids = set()
        self.rewards_episode= []
        
        self.discount = discount
        self.epi_length = epi_length
        self.iterations = iterations
        self.theta = theta
        
        self.goal = (48,12)
        
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
        
        self.visited_states = {}
        self.frequency_action = {}
        
        self.state_action_state = {}
        self.reward_state_action_state = {}
        
        self.probabilities = {}
        self.actual_rewards = {}
        
        self.means = []
        self.variances = []
        
        
        
        for i in range(0,50):
            for j in range(0,25):
                state = (i,j)
                self.visited_states[state] = 0
                for act in self.moves:
                    self.frequency_action[(state,act)] = 0
                    
                    for k in range(0,50):
                        for l in range(0,25):
                            state_2 = (k,l)
                            self.state_action_state[(state,act,state_2)] = 0
                            self.reward_state_action_state[(state,act,state_2)] = 0
                            
                    
                    
    ### This is something we know, robot doesn't.
    def actual_action(self,action):
        
        prob_move = []
        for i in range(0,4):
            if self.moves[i] == action :
                prob_move.append(0.8)
            else :
                prob_move.append((0.2)/3)
        prob_move = np.array(prob_move)
        return np.random.choice(self.moves,p=prob_move)
        
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
                self.visited_states[state] = self.visited_states[state] + 1
                break
            
            action_list = []
            
            if self.visited_states[state] == 0:
                self.visited_states[state]  = 1
                for act in self.moves :
                    action_list.append(act)
            
            else :
                self.visited_states[state]  += 1
                visits = min([ self.frequency_action[(state,act)] for act in self.moves])
                for act in self.moves :
                    if self.frequency_action[(state,act)] == visits:
                         action_list.append(act)
                         
            action = np.random.choice(action_list)
            self.frequency_action[(state,action)] = self.frequency_action[(state,action)] + 1
            
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
            reward = reward_history[-1]
            
            self.state_action_state[(state,action,new_state)] = self.state_action_state[(state,action,new_state)] + 1
            self.reward_state_action_state[(state,action,new_state)] =self.reward_state_action_state[(state,action,new_state)] + reward
            
            (x,y) = new_state
            
        arr1 = []
        for i in range(0,50):
            for j in range(0,25):
                if(i,j) in self.grids:
                    continue
                arr1.append(self.visited_states[(i,j)])
        arr1 = np.array(arr1)
        self.means.append(np.mean(arr1))
        self.variances.append(np.std(arr1))
        
        
    def estimate_parameters(self):
        
        for (state1,act,state2) in self.state_action_state:
            
            if self.state_action_state[(state1,act,state2)] == 0:
                continue
            self.probabilities[(state1,act,state2)] = self.state_action_state[(state1,act,state2)]/self.frequency_action[(state1,act)]
        for (state1,act,state2) in self.reward_state_action_state:
            if self.state_action_state[(state1,act,state2)] == 0:
                continue
            self.actual_rewards[(state1,act,state2)] = self.reward_state_action_state[(state1,act,state2)]/self.state_action_state[(state1,act,state2)]
    
    
    def updated_value(self,position,action,values):
        
        actual_val = 0
        state = position
        allowed_positions = set()
        for act in self.moves:
            new_pos = add_dir(position,self.move_shift[act])
            if new_pos in self.grids:
                new_pos = position
            if (position,action,new_pos) in self.probabilities:
                allowed_positions.add(new_pos)
        allowed_positions = list(allowed_positions)
        
        for new_pos in allowed_positions:
             val =  self.reward_state_action_state[(position,action,new_pos)] + (self.discount* values[new_pos[0]][new_pos[1]])
             actual_val += val* self.probabilities[(position,action,new_pos)]
             
        return actual_val
            
    
    def value_iteration_estimated(self):    
        values = np.zeros((50,25))
        for iter1 in range(0,self.iterations):
            
            new_values = np.zeros((50,25))
            
            for i in range(0,50):
                for j in range(0,25):
                    if (i,j) in self.grids:
                        continue
                    pref_action = None
                    val = -9999999.99
                    for act in range(0,4):
                        store = self.updated_value((i,j),self.moves[act],values)
                        
                        if store > val:
                            val = store
                            pref_action = self.moves[act]
                            
                    new_values[i][j] = val
                    self.optimal_policy[i][j] = pref_action
                    
            diff1 = np.max(np.abs(new_values-values))
            values = new_values
            if diff1 <= self.theta:
                break
        
        self.values_obtained = values
    
        
grid1 = MDP_grid(0.99,1000,100,0.1)
for i in range(0,100):
    grid1.create_episode()
     

x_axis = [i for i in range(1,101)]           
plt.xlabel("Episodes")
plt.errorbar(x_axis, grid1.means, grid1.variances)
plt.savefig("a.png")
plt.show()

grid1.estimate_parameters()
grid1.value_iteration_estimated()

def arrow_coordinates(i, j, action):
    if action == 'UP':
        return i, j + 0.2, 0, 0.6
    elif action == 'DOWN':
        return i, j - 0.2, 0, -0.6
    elif action == 'LEFT':
        return i - 0.2, j, -0.6, 0
    elif action == 'RIGHT':
        return i + 0.2, j, 0.6, 0
                        
                    
all_arrows = []
for i in range(50):
    for j in range(25):
        if grid1.optimal_policy[i][j] != None:
            x, y, dx, dy = arrow_coordinates(i, j, grid1.optimal_policy[i][j])
            all_arrows.append((x, y, dx, dy))
            
            
##Finally the plotting starts
            
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
plt.savefig('c.png', bbox_inches='tight')
plt.show()
            
        
