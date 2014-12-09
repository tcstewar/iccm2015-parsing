import numpy as np


k = 8
M = 50000000000000
q = 0.05


D = 3.16*(k-0.25)* np.log(M/(q**3))

print D

