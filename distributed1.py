import nengo
import nengo.spa
import numpy as np

D = 128
N = 30

rng = np.random.RandomState(seed=2)
vocab = nengo.spa.Vocabulary(D, rng=rng)

class LinearRule(object):
    def __init__(self, vocab, pre, post, gain=2, min=0, control_inhibit=False):
        self.gain = 2
        self.min = 0
        self.pre = vocab.parse(pre).v
        self.post = vocab.parse(post).v
        self.control_inhibit = control_inhibit
    def apply(self, state):
        x = np.dot(state, self.pre)
        if x < self.min:
            x = self.min
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
        if x < 0.5:
            x = 0
        v = x * self.gain
        #if self.control_inhibit:
        #    if np.dot(state, self.post) > 0:
        #        v = 0
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

    
nouns = ['DOG', 'CAT', 'N0', 'N1'][:2]
verbs = ['CHASE', 'V0', 'V1'][:1]

rules = RuleSet(vocab, tau=0.5, gain=1.00)

for noun in nouns:
    rules.add(LinearRule(vocab, noun, 'N_'+noun))
    rules.add(LinearRule(vocab, 'N_'+noun, '-0.2*'+noun, control_inhibit=True))
    
    rules.add(LinearRule(vocab, 'N_'+noun, 'NP_'+noun))
    rules.add(LinearRule(vocab, 'NP_'+noun, '-0.2*N_'+noun, control_inhibit=True))
for verb in verbs:
    rules.add(LinearRule(vocab, verb, 'V_'+verb))
    rules.add(LinearRule(vocab, 'V_'+verb, '-0.2*'+verb, control_inhibit=True))
    
    for noun in nouns:
        rules.add(PairRule(vocab, 'V_'+verb, 'N_'+noun, '2*VP_'+verb+'_'+noun))
        rules.add(LinearRule(vocab, 'VP_'+verb+'_'+noun, '-0.2*N_'+noun, control_inhibit=True))
        rules.add(LinearRule(vocab, 'VP_'+verb+'_'+noun, '-0.2*V_'+verb, control_inhibit=True))
    

model = nengo.Network(seed=1)
model.config[nengo.Ensemble].neuron_type=nengo.LIFRate()
with model:
    state = nengo.Ensemble(n_neurons=N*D, dimensions=D, radius=10)
    
    
    nengo.Connection(state, state, synapse=0.1, function=rules.apply)
    
    def stim_func(t):
        stim = ['DOG', 'CHASE', 'CAT', '0', '0', '0', '0', '0']
        index = int(t / 0.5) % len(stim)
        return vocab.parse(stim[index]).v
        
    stimulus = nengo.Node(stim_func)
    nengo.Connection(stimulus, state, transform=0.3)
    
    probe = nengo.Probe(state, synapse=0.03)
    
sim = nengo.Simulator(model, seed=1)
sim.run(4)

print vocab.text(sim.data[probe][-1], maximum_count=3)

import pylab
pylab.figure(figsize=(10,7))
lines = pylab.plot(sim.trange(), vocab.dot(sim.data[probe].T).T, linewidth=5)
pylab.legend(lines, vocab.keys, loc='best')
pylab.show()
