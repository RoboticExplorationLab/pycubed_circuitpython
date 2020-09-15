import math
import ulab as np
import time

def norm(x):
    """norm of an array"""
    return math.sqrt(np.numerical.sum(x**2))

def normalize(x):
    """normalize an array (not in place)"""
    return x/norm(x)

def normalize_ip(x):
    """normalize an array in place"""
    x = x/norm(x)

def changeit(a):
    a[2] = 69.0

#a = np.array([1,2,3.0])
#b = np.array([4,3.0,-7.6])

#print(normalize(a))

#normalize_ip(a)

#print(a)

a1 = np.array([1,2,3])
a2 = [1.0,2.0,3.0]

changeit(a1)
changeit(a2)

print(a1)
print(a2)

N = 1000
n = 100

t1 = time.monotonic()
for i in range(N):
    x = [0.0 for _ in range(n)]
print(time.monotonic()-t1)

t1 = time.monotonic()
for i in range(N):
    x = [0.0]*n
print(time.monotonic()-t1)