import nengo
import nengo.spa
import numpy as np

D = 128
N = 30

rng = np.random.RandomState(seed=5)
vocab = nengo.spa.Vocabulary(D, rng=rng)

class LinearRule(object):
    def __init__(self, vocab, pre, post, gain=2, min=0, control_inhibit=False):
        self.gain = gain
        self.min = 0
        self.pre = vocab.parse(pre).v
        self.post = vocab.parse(post).v
        self.control_inhibit = control_inhibit
    def apply(self, state):
        x = np.dot(state, self.pre)
        #if x < self.min:
        #    x = self.min
        v = x * self.gain
        if self.control_inhibit:
            if np.dot(state, self.post) > 0:
                v = 0
        return v * self.post    
    
class PairRule(object):
    def __init__(self, vocab, pre1, pre2, post, gain=2, min=0, control_inhibit=False):
        self.gain = 2
        self.min = 0
        self.pre1 = vocab.parse(pre1).v
        self.pre2 = vocab.parse(pre2).v
        self.post = vocab.parse(post).v
        self.control_inhibit = control_inhibit
    def apply(self, state):
        #x = np.dot(state, self.pre1) * np.dot(state, self.pre2)
        x = min(np.dot(state, self.pre1), np.dot(state, self.pre2))
        if x < self.min:
            x = self.min
        v = x * self.gain
        if self.control_inhibit:
            if np.dot(state, self.post) > 0:
                v = 0
        return v * self.post    
    

    
    
class RuleSet(object):
    def __init__(self, vocab, tau, gain):
        self.vocab = vocab
        self.tau = tau
        self.gain = gain
        self.rules = []
    def add(self, rule):
        self.rules.append(rule)
        
    def apply(self, x):
        delta = np.zeros_like(x)
        for r in self.rules:
            delta += r.apply(x)
        v = x * self.gain + self.tau * delta

        #norm = np.linalg.norm(v)
        #if norm > 3:
        #    v = v / norm * 3
        return v


det_paths = ['SL_NPL_DET','SM_NPL_DET','SR_VPR_NPL_DET']
n_paths = ['SL_NPR_N','SM_NPR_N','SR_VPR_NPR_N']
v_paths = ['SR_VPM_V','SR_VPL_V']
aux_paths = ['SL_AUX']

pos_constraints = [(det_paths[0],n_paths[0]),
                   (det_paths[1],n_paths[1]),
                   (det_paths[1],aux_paths[0]),
                   (det_paths[2],n_paths[2]),
                   (det_paths[2],v_paths[1]),
                   (n_paths[1],aux_paths[0]),
                   (n_paths[2],v_paths[1])]

neg_constraints = [(det_paths[0],det_paths[1]),
                   (det_paths[0],n_paths[1]),
                   (det_paths[0],aux_paths[0]),
                   (det_paths[1],n_paths[0]),
                   (det_paths[2],v_paths[0]),
                   (n_paths[0],n_paths[1]),
                   (n_paths[0],aux_paths[0]),
                   (n_paths[2],v_paths[0]),
                   (v_paths[1],v_paths[0])]


rules = RuleSet(vocab, tau=0.2, gain=0)

for pre, post in pos_constraints:
    rules.add(LinearRule(vocab, pre, '2*'+post))
for pre, post in neg_constraints:
    rules.add(LinearRule(vocab, pre, '-2*'+post))


model = nengo.Network(seed=1)
model.config[nengo.Ensemble].neuron_type=nengo.LIFRate()
with model:
    state = nengo.Ensemble(n_neurons=N*D, dimensions=D, radius=2)
    
    
    nengo.Connection(state, state, synapse=0.1, function=rules.apply)
    
    def stim_func(t):
        stim = ['SL_NPR_N+SR_VPL_V'] * 10
        stim = ['SM_NPL_DET+SR_VPM_V'] * 10
        index = int(t / 0.5) % len(stim)
        return vocab.parse(stim[index]).v
        
    stimulus = nengo.Node(stim_func)
    nengo.Connection(stimulus, state, transform=0.3)
    
    probe = nengo.Probe(state, synapse=0.03)
    
sim = nengo.Simulator(model, seed=1)
sim.run(1)

print vocab.text(sim.data[probe][-1], maximum_count=6, minimum_count=6)

import pylab
pylab.figure(figsize=(14,7))
pylab.axes((0.08, 0.1, 0.87, 0.85))
colors = ['k', '#555555', '#aaaaaa']
linestyles = ['-', '--', '-.', ':']
d = vocab.dot(sim.data[probe].T)
lines = []
for i, row in enumerate(d):
    line, = pylab.plot(sim.trange(), row, color=colors[i%len(colors)],
                                     linestyle=linestyles[i%len(linestyles)],
                                     linewidth=5)
    lines.append(line)

keys = list(vocab.keys)
keymap = {
    'SL_NPR_N': 'S(NP(?,N),?)',
    'SM_NPL_DET': 'S(?,NP(DET,?))',
    'SR_VPM_V': 'S(?,VP(?,V,?))',
    'SR_VPL_V': 'S(?,VP(V,?))',
    'SL_AUX': 'S(AUX,?)',
    'SM_NPR_N': 'S(?,NP(?,N),?)',
    'SL_NPL_DET': 'S(NP(DET,?))',
    'SR_VPR_NPL_DET': 'S(?,VP(?,NP(DET,?))',
    'SR_VPR_NPR_N': 'S(?,VP(?,NP(?,N))',
    }
for i,k in enumerate(keys):
    keys[i] = keymap.get(k,k)
pylab.legend(lines, keys, loc='center right')
#pylab.text(0.25, -0.7, 'DOG', ha='center', va='center', fontsize=20)
#pylab.text(0.75, -0.7, 'CHASE', ha='center', va='center', fontsize=20)
#pylab.text(1.25, -0.7, 'CAT', ha='center', va='center', fontsize=20)
#pylab.ylim(-1, 7)
pylab.xlabel('time (s)')
pylab.ylabel('similarity (dot product)')
pylab.savefig('complete.png', dpi=300)
pylab.show()
