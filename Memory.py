import random
import numpy as np
class Memory:
    def __init__(self, max_memory, single_access=False, use_PER=False, hyperparams=(None, None, None)):
        self.max_memory = max_memory
        self.single_access = single_access # if True, remove samples once they've been accessed
        self.samples = []
        self.use_PER = use_PER
        self.e, self.a, self.b = hyperparams
        if self.use_PER:
            self.tree = SumTree(max_memory)

    def add_sample(self, sample):
        if self.use_PER:
            max_priority = np.amax(self.tree.tree[-self.tree.capacity:])
            # avoid having having max_priority be 0
            if max_priority == 0:
                max_priority = 1
            self.tree.add(max_priority, sample)
        else:
            self.samples.append(sample)
            
        if len(self.samples) > self.max_memory:
            self.samples.pop(0)

    def sample(self, num_samples):
        if self.num_experiences_stored < num_samples:
            return self.sample(self.num_experiences_stored)
        elif num_samples == 0:
            if self.use_PER:
                return [], []
            else:
                return []
        
        if self.use_PER:
            sample = []
            priority_segment = self.tree.total_priority / num_samples
            
            idxs, priorities, data = zip(*[self.tree.get_leaf(random.uniform(priority_segment * i, priority_segment * (i+1))) for i in range(num_samples)])
            filtered_data = list(map(lambda x: x if isinstance(x, tuple)
                                     else random.choice(self.tree.data), data))
            
            return idxs, filtered_data
        else:
            sample = []
            sample = random.sample(self.samples, num_samples)
            
            if self.single_access:
                for i in sample:
                    self.samples.remove(i)
            return sample
    
    def update(self, idxs, errors):
        errors += self.e
        errors = np.minimum(errors, 1)
        for idx, p in zip(idxs, np.power(errors, self.a)):
            self.tree.update(idx, p)
    
    @property
    def num_experiences_stored(self):
        return self.tree.data_counter if self.use_PER else len(self.samples)
    
class SumTree:
    data_counter = 0
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * self.capacity - 1)
        self.data = np.zeros(self.capacity, dtype=tuple)
    
    def add(self, priority, data):
        # add a node to the tree
        tree_idx = self.data_counter + self.capacity - 1
        self.data[self.data_counter] = data
        self.update(tree_idx, priority)
        self.data_counter += 1
        if self.data_counter >= self.capacity:
            # start overwriting if we're out of room
            self.data_counter = 0
    
    def update(self, tree_idx, priority):
        # update new node
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        
        # update the rest of the nodes
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change
    
    def get_leaf(self, value):
        # returns (leaf index, priority value, experience)
        parent_idx = 0
        leaf_idx = 0
        
        while True:
            left_idx = 2 * parent_idx + 1
            right_idx = left_idx + 1
            
            if left_idx >= len(self.tree):
                # end the search if this is the bottom
                leaf_idx = parent_idx
                break
            else:
                # otherwise, go down a layer and keep searching
                if value <= self.tree[left_idx]:
                    parent_idx = left_idx
                else:
                    value -= self.tree[left_idx]
                    parent_idx = right_idx
        
        data_idx = leaf_idx - self.capacity + 1
        
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]
    
    @property
    def total_priority(self):
        return self.tree[0]