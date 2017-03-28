import math
p = input("Enter p -> ")
s = input ("Enter s -> ")
p = float(p)
s = float(s)
entropy = s * (1 - p * p - (1-p) * (1-p))
print (entropy)
    
    