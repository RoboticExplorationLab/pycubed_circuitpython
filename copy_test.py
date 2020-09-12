
import time
import board
import ulab as np
import math

# create np.array
a = np.zeros(100)

# reset array function by looping through
def reset_array(input_array):
    for i in range(len(input_array)):
        input_array[i] = 0.0



# time this for 1000 runs
t1 =time.monotonic()
for i in range(1000):
    reset_array(a)

print("Loop through time",time.monotonic()-t1)


t2 = time.monotonic()
for i in range(1000):
    a = a*0.0

print("Multiply time",time.monotonic()-t2)