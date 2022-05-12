from math import sqrt, pi, sin, cos
a = 279
e = 131

da = e / sqrt(3) + 2/3*sqrt(2)* e
ae = a+da

print(ae)

h_inscr = sqrt(3) / 2 * a
h_conscr = sqrt(3) / 2 * ae

area_i = a * h_inscr / 2
area_c = ae * h_conscr / 2

print(area_i/area_c)
print(e / a)
print(h_conscr / ae)