
import numpy as np
from roadsimulator.colors import White
from roadsimulator.layers.layers import Background, DrawLines, Perspective, Crop
from roadsimulator.simulator import Simulator

simulator = Simulator()
white = White()

simulator.add(Background(n_backgrounds=3, path='./ground_pics', input_size=(250, 200)))
simulator.add(DrawLines(input_size=(250, 200), color_range=white))
simulator.add(Perspective())
simulator.add(Crop())

simulator.generate(n_examples=1000, path='data')


