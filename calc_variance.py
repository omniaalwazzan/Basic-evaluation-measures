# this to calculate the v

import math

N = 6

# F1 - Micro 
f1 = 0.812
f2 = 0.801
f3 = 0.781
f4 = 0.723
f5 = 0.740
f6 = 0.715


total_fMic = ( f1+ f2+f3+f4+f5 +f6)/N

# F1 - Mac
f1 = 0.653
f2 = 0.612
f3 = 0.601
f4 = 0.665
f5 = 0.742
f6 = 0.678

total = ( f1+ f2+f3+f4+f5 +f6)/N


x1= f1- total
x2= f2- total
x3= f3- total
x4= f4- total
x5= f5- total
x6= f6- total

def number(a):
    return a*a

x1 = number(x1)
x2 = number(x2) 
x3 = number(x3) 
x4 = number(x4) 
x5 = number(x5) 
x6 = number(x6) 

out = x1+x2+x3+x4+x5+x6

variance = out/(N-1)
print(variance)
std = math.sqrt(out)
print(std)

