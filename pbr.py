import time
import board
import math
import ulab as np


def foo(a):
    a[1,1] = 69


a = np.eye(3)
#a = [1,2,3]

foo(a)

print(a)