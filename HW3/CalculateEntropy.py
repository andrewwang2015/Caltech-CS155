import math
p = input("Enter p -> ")
s = input ("Enter s -> ")
p = float(p)
s = float(s)
if p == 0:
    firstTerm = 0
else:
    firstTerm = p * math.log(p,2)
    
if p == 1:
    secondTerm = 0
else:
    secondTerm = (1-p) * math.log(1-p, 2)
entropy = - s * (firstTerm + secondTerm)
print (entropy)
    
    