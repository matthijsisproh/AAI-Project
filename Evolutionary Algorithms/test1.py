import math




delta_o = math.tanh(0.46) * (0 - 0.44)
delta_h = math.tanh(-0.2) * 0.6 * delta_o
delta_l = math.tanh(0.8) * 0.9 * delta_o
print(delta_h)



w = 0.6 + 0.1 * -0.2 * -0.26
print(w)


calc = (-0.1) * 0.43
print(calc)