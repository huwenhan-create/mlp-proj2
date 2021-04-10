import numpy as np

def entropyValue(prob):
    if (prob==0 or prob==1):
        entropy = 0
    else:
        entropy = -(prob*np.log2(prob)+(1-prob)*np.log2(1-prob))
    return entropy