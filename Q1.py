import matplotlib.pyplot as plt
import numpy as np
import matplotlib.markers as markers
import copy


def add_dir(cell,direction):
    return (cell[0]+direction[0],cell[1]+direction[1])

class MDP_grid:
    
    def __init__(self,discount,iterations,theta):
        
        self.optimal_policy = [[None for j in range(25)] for i in range(50)]
        self.values_obtained_list = []
        self.optimal_policy_list = {}
        self.grids = set()
        self.diff_list = []
        
        self.discount = discount
        self.iterations = iterations
        self.theta = theta
        
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
    
        

    def updated_value(self,position,direction,values):
        
        prob_move = []
        for i in range(0,4):
            if self.moves[i] == direction :
                prob_move.append(0.8)
            else :
                prob_move.append((0.2)/3)
        
        actual_val = 0
                 
        ## Now writing the required equation:
        for i in range(0,4):
            reward = None
            new_position = add_dir(position,self.move_shift[self.moves[i]])
            if new_position in self.grids:
                new_position = position
            if new_position == self.goal:
                reward = 100
            elif new_position == position:
                reward = -1
            else :
                reward = 0
                
            val = reward + (self.discount* values[new_position[0]][new_position[1]])
            actual_val += val* prob_move[i]
            
        return actual_val
    
    
    def value_iteration(self):
        
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
            self.optimal_policy_list[iter1]= copy.deepcopy(self.optimal_policy)
            self.values_obtained_list.append(copy.deepcopy(new_values))
            self.diff_list.append(diff1)
            if diff1 <= self.theta:
                break
        
        self.values_obtained = values
 
    
    
grid1 = MDP_grid(0.1,100,0.1)
grid1.value_iteration()

        
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
plt.savefig('a.png', bbox_inches='tight')
plt.show()
  

grid2 = MDP_grid(0.99,100,0.1)
grid2.value_iteration()          

##Epoch no 20

all_arrows_2 = []
for i in range(50):
    for j in range(25):
        if grid2.optimal_policy_list[19][i][j] != None:
            x, y, dx, dy = arrow_coordinates(i, j, grid2.optimal_policy_list[19][i][j])
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
X = grid2.values_obtained_list[19].reshape(-1, 1)
x  = (X - np.min(X))/ (np.max(X)- np.min(X)) * 255
scat1 = plt.scatter(all_points[:, 1], all_points[:, 0], s=200, c=X[:, 0], cmap='Greys', edgecolors='k', marker=marker)
scat2 = plt.scatter([x[0] for x in all_walls], [x[1] for x in all_walls], s=200, c=wall_color, edgecolors='k', marker=marker)
for i in range(len(all_arrows)):
    plt.arrow(all_arrows_2[i][0], all_arrows_2[i][1], all_arrows_2[i][2], all_arrows_2[i][3], length_includes_head=True, head_width=0.15, edgecolor='b', facecolor='y')
plt.savefig('b_20.png', bbox_inches='tight')
plt.show()

##Epoch 50

all_arrows_3 = []
for i in range(50):
    for j in range(25):
        if grid2.optimal_policy_list[49][i][j] != None:
            x, y, dx, dy = arrow_coordinates(i, j, grid2.optimal_policy_list[49][i][j])
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
X = grid2.values_obtained_list[49].reshape(-1, 1)
x  = (X - np.min(X))/ (np.max(X)- np.min(X)) * 255
scat1 = plt.scatter(all_points[:, 1], all_points[:, 0], s=200, c=X[:, 0], cmap='Greys', edgecolors='k', marker=marker)
scat2 = plt.scatter([x[0] for x in all_walls], [x[1] for x in all_walls], s=200, c=wall_color, edgecolors='k', marker=marker)
for i in range(len(all_arrows)):
    plt.arrow(all_arrows_3[i][0], all_arrows_3[i][1], all_arrows_3[i][2], all_arrows_3[i][3], length_includes_head=True, head_width=0.15, edgecolor='b', facecolor='y')
plt.savefig('b_50.png', bbox_inches='tight')
plt.show()

##Epoch 100

all_arrows_4 = []
for i in range(50):
    for j in range(25):
        if grid2.optimal_policy_list[99][i][j] != None:
            x, y, dx, dy = arrow_coordinates(i, j, grid2.optimal_policy_list[99][i][j])
            all_arrows_4.append((x, y, dx, dy))                        
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
X = grid2.values_obtained_list[99].reshape(-1, 1)
x  = (X - np.min(X))/ (np.max(X)- np.min(X)) * 255
scat1 = plt.scatter(all_points[:, 1], all_points[:, 0], s=200, c=X[:, 0], cmap='Greys', edgecolors='k', marker=marker)
scat2 = plt.scatter([x[0] for x in all_walls], [x[1] for x in all_walls], s=200, c=wall_color, edgecolors='k', marker=marker)
for i in range(len(all_arrows)):
    plt.arrow(all_arrows_4[i][0], all_arrows_4[i][1], all_arrows_4[i][2], all_arrows_4[i][3], length_includes_head=True, head_width=0.15, edgecolor='b', facecolor='y')
plt.savefig('b_100.png', bbox_inches='tight')
plt.show()

### Now plotting error as required.

iter_num = 10
grid3 = MDP_grid(0.01,iter_num,0.1)
grid3.value_iteration()

error_low = grid3.diff_list
len1 = len(error_low)
for i in range(len1,10):
    error_low.append(0)
error_high = grid2.diff_list

x_axis = [i for i in range(0,iter_num)]
plt.xlabel("Iterations")
plt.ylabel("Max Norm Error")
plt.plot(x_axis,error_low,color="red",label="discount=0.01")
plt.legend()
plt.savefig("c_1.png")
plt.show()

x_axis = [i for i in range(0,100)]
plt.xlabel("Iterations")
plt.ylabel("Max Norm Error")
plt.plot(x_axis,error_high,color="blue",label="discount=0.99")
plt.legend()
plt.savefig("c_2.png")
plt.show()

    
