import numpy as np

class Lake():
    
    def __init__(self, width, height, start):
        self.width = width
        self.height = height
        self.i = start[0]
        self.j = start[1]

    def set(self, rewards, actions):
        # rewards should be a dict of: (i, j): r (row, col): reward
        # actions should be a dict of: (i, j): A (row, col): list of possible actions
        self.rewards = rewards
        self.actions = actions

    def set_state(self, s):
        self.i = s[0]
        self.j = s[1]

    def current_state(self):
        return (self.i, self.j)

    def is_terminal(self, s):
        return s not in self.actions

    def move(self, action):
        # check if legal move first
        if action in self.actions[(self.i, self.j)]:
            if action == 'U':
                self.i -= 1
            elif action == 'D':
                self.i += 1
            elif action == 'R':
                self.j += 1
            elif action == 'L':
                self.j -= 1
            # return a reward (if any)
        return self.rewards.get((self.i, self.j), 0)

    def undo_move(self, action):
        # these are the opposite of what U/D/L/R should normally do
        if action == 'U':
            self.i += 1
        elif action == 'D':
            self.i -= 1
        elif action == 'R':
            self.j -= 1
        elif action == 'L':
            self.j += 1
        # raise an exception if we arrive somewhere we shouldn't be
        # should never happen
        assert(self.current_state() in self.all_states())

    def game_over(self):
        # returns true if game is over, else false
        # true if we are in a state where no actions are possible
        return (self.i, self.j) not in self.actions

    def all_states(self):
        # possibly buggy but simple way to get all states
        # either a position that has possible next actions
        # or a position that yields a reward
        return set(self.actions.keys()) | set(self.rewards.keys())
SMALL_ENOUGH = 1e-3
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')
if __name__ == '__main__':
  grid = Lake(4,4,(0,0))
  rewards = {(3, 3): 1, (1, 1): -1, (1, 3): -1, (2, 0): -1, (3, 2): -1}
  actions = {
    (0, 0): ('D', 'R'),
    (0, 1): ('L', 'D', 'R'),
    (0, 2): ('L', 'D', 'R'),
    (0, 3): ('L', 'D'),
    (1, 0): ('U', 'D', 'R'),
    (1, 2): ('U', 'D', 'R', 'L'),
    (2, 1): ('U', 'D', 'R', 'L'),
    (2, 2): ('U', 'D', 'R', 'L'),
    (2, 3): ('U', 'D', 'L'),
    (3, 0): ('U', 'R'),
    (3, 1): ('U', 'R', 'L')
}
  grid.set(rewards, actions)
  
  print("rewards:")
  print_values(grid.rewards, grid)

  
  policy = {}
  for s in grid.actions.keys():
    policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

  # initial policy
  print("initial policy:")
  print_policy(policy, grid)

  # initialize V(s)
  V = {}
  states = grid.all_states()
  for s in states:
    # V[s] = 0
    if s in grid.actions:
      V[s] = np.random.random()
    else:
      # terminal state
      V[s] = 0

  # repeat until convergence - will break out when policy does not change
  while True:

    # policy evaluation step
    while True:
      biggest_change = 0
      for s in states:
        old_v = V[s]

        # V(s) only has value if it's not a terminal state
        new_v = 0
        if s in policy:
          for a in ALL_POSSIBLE_ACTIONS:
            if a == policy[s]:
              p = 0.5
            else:
              p = 0.5/3
            grid.set_state(s)
            r = grid.move(a)
            new_v += p*(r + GAMMA * V[grid.current_state()])
          V[s] = new_v
          biggest_change = max(biggest_change, np.abs(old_v - V[s]))

      if biggest_change < SMALL_ENOUGH:
        break

    # policy improvement step
    is_policy_converged = True
    for s in states:
      if s in policy:
        old_a = policy[s]
        new_a = None
        best_value = float('-inf')
        # loop through all possible actions to find the best current action
        for a in ALL_POSSIBLE_ACTIONS: # chosen action
          v = 0
          for a2 in ALL_POSSIBLE_ACTIONS: # resulting action
            if a == a2:
              p = 0.5
            else:
              p = 0.5/3
            grid.set_state(s)
            r = grid.move(a2)
            v += p*(r + GAMMA * V[grid.current_state()])
          if v > best_value:
            best_value = v
            new_a = a
        policy[s] = new_a
        if new_a != old_a:
          is_policy_converged = False

    if is_policy_converged:
      break

  print("values:")
  print_values(V, grid)
  print("policy:")
  print_policy(policy, grid)
