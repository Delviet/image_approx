import cv2
import numpy as np
from copy import deepcopy
from tqdm import tqdm

path = './init.jpg' # path to an image

img = cv2.imread(path)
required_size = tuple(list(img.shape[:2]) + [1])
option_num = sum(required_size[:-1]) * 2 - 4 # number of the edge pixels

# turning an image into special grayscale
min_img = img.min(axis=2, keepdims = True)
img = min_img
cv2.imwrite(f'./ideal.png', img)

white = img.mean() > 127.5 # makes initial guess preciser

# edgepoint id -> (y, x) mapping, to be used in Chromosomes
h, w, _ = required_size
mapper = []
for i in range(w):
    mapper.append((i, 0))
for i in range(1, h):
    mapper.append((w-1, i))
for i in range(w-2, 0, -1):
    mapper.append((i, h-1))
for i in range(h-1, 0, -1):
    mapper.append((0, i))

class Chromosome:
    def __init__(self):
        self.init_point = np.random.randint(0, option_num) # choice of the segment endpoints
        self.final_point = np.random.randint(0, option_num)
        self.color = np.random.randint(0, 255, 1).tolist() #such a form for the library compatibility

    def mutate(self): 
        self.init_point = (self.init_point + np.random.randint(-10, 11)) % option_num #ensure that id in valid range
        self.final_point = (self.final_point + np.random.randint(-10, 11)) % option_num   
        
        rand_val = np.random.randint(-5, 6)
        self.color[0] = int(np.clip(self.color[0] + rand_val, 0, 255))
    
    def _process(self, point_idx): #turn edgepoint idx into coordinates of the point
        return mapper[point_idx]

    def display(self): #useful view of the core information
        return (self._process(self.init_point), 
                self._process(self.final_point), 
                self.color)

    def __str__(self):
        i_p, f_p, c = self.display()
        return f"Chromosome of a line starting at point {i_p} and ending in {f_p} having color {c}"

class Sample:
    def __init__(self, chromo_num):
        self.chromo_list = [] # list of chromosomes
        self.init = np.zeros(required_size) + 255*white # initial representation
        for i in range(chromo_num):
            self.chromo_list.append(Chromosome()) 
        self.cache = self.init
        self.add = False
    
    def display(self): # create an image
        if self.add: # add only the last one chromosome
            temp_img = self.cache
            chromo = self.chromo_list[-1]
            i_p, f_p, c = chromo.display()
            cv2.line(temp_img, i_p, f_p, list(c))
            self.add = False
        else: # if deleted or mutated - fully recreate
            temp_img = self.init # create a background depending on image color
            for chromo in self.chromo_list:
                i_p, f_p, c = chromo.display()
                cv2.line(temp_img, i_p, f_p, list(c))
        self.cache = temp_img
        return temp_img

    def evaluate(self, target_image): # check the distance
        existing = self.display()
        return ((existing - target_image)**2).sum() + len(self.chromo_list)**2

    def _generate_mask(self, proba, size = (1,)):
        return np.random.choice(2, size = size, p = [1 - proba, proba])

    def _change_chrom(self):
        mask = np.random.choice(len(self.chromo_list))
        self.chromo_list[mask].mutate()

    def _new_chroms(self):
        self.chromo_list.append(Chromosome())

    def _del_chroms(self):
        if len(self.chromo_list) < 10:
            pass
        else: 
            to_del = np.random.choice(len(self.chromo_list))
            del self.chromo_list[to_del]

    def mutate(self, timestamp):
        if timestamp < 0.5:
            start = timestamp * 2
            probs = [0.4 * start, 1.0 - 0.7 * start, 0.3 * start]
        else:
            start = (timestamp - 0.5) * 2
            probs = [0.4 + 0.3 * start, 0.3 - 0.1 * start, 0.3 - 0.2 * start]
        name = np.random.choice(['change', 'new', 'del'], p = probs)
        if name == 'change':
            self._change_chrom()
        elif name == 'new':
            self.add = True
            self._new_chroms()
        elif name == 'del':
            self._del_chroms()

my_samp = Sample(1)
best_samp = None
best_score = 10**100
iter_num = 1e6
for i in tqdm(range(iter_num)): 
    curr_score = my_samp.evaluate(img)
    if curr_score < best_score:
        best_samp = deepcopy(my_samp)
        best_score = curr_score
    else:
        my_samp = deepcopy(best_samp)
    if i % 1000 == 0:
        cv2.imwrite(f'./iter_{i}.png', best_samp.display())
    my_samp.mutate(i/iter_num)