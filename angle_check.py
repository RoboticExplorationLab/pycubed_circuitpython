from math import cos
from math import sin



# s
t = 3622269


w = 7.292115146706979e-5

w1 = 7.29211e-5
w2 = 5.14671e-11
#w3 = 6.979e-17

print(t)
print(sin(t*w))

a = t*w1
b = t*w2


print(sin(a)*cos(b) + cos(a)*sin(b))

#c = t*w3

#print(cos(b)*cos(c)*sin(a) + cos(a)*cos(c)*sin(b) + cos(a)*cos(b)*sin(c) - sin(a)*sin(b)*sin(c))

print(cos(w*t))
print(cos(a)*cos(b) - sin(a)*sin(b))