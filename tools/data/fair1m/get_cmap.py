import matplotlib.pyplot as plt
import numpy as np

CLASSES = ('A220', 'A321', 'A330', 'A350', 'ARJ21', 'Baseball Field',
           'Basketball Court', 'Boeing737', 'Boeing747', 'Boeing777',
           'Boeing787', 'Bridge', 'Bus', 'C919', 'Cargo Truck',
           'Dry Cargo Ship', 'Dump Truck', 'Engineering Ship',
           'Excavator', 'Fishing Boat', 'Football Field', 'Intersection',
           'Liquid Cargo Ship', 'Motorboat', 'Passenger Ship', 'Roundabout',
           'Small Car', 'Tennis Court', 'Tractor', 'Trailer', 'Truck Tractor',
           'Tugboat', 'Van', 'Warship', 'other-airplane', 'other-ship',
           'other-vehicle')


def generate_palette(n_colors):
    cmap = plt.get_cmap('nipy_spectral', n_colors)
    palette = [cmap(i) for i in range(cmap.N)]
    palette = [(int(r*255), int(g*255), int(b*255)) for r, g, b, _ in palette]
    return palette


PALETTE = generate_palette(len(CLASSES))
print("PALETTE = [", end="")
for i, color in enumerate(PALETTE):
    if i != len(PALETTE) - 1:
        print(str(color) + ", ", end="")
    else:
        print(str(color), end="")
print("]")
