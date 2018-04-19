# -*- coding: utf-8 -*-
# from HW_3.cars import *
from cars.world import SimpleCarWorld
from cars.agent import SimpleCarAgent
from cars.physics import SimplePhysics
from cars.track import generate_map
import numpy as np
import random

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--iterations", type=int)
parser.add_argument("-f", "--filename", type=str)
parser.add_argument("-e", "--evaluate", type=bool)
parser.add_argument("--seed", nargs='+', type=int)
args = parser.parse_args()

print(args.iterations, args.seed, args.filename, args.evaluate)

# steps = args.steps
iterations = args.iterations

seeds = args.seed if args.seed else [23]
# np.random.seed(seed)
# random.seed(seed)
# m = generate_map(8, 5, 3, 3)


test_seeds = [3,13,18]

etas = [0.02, 0.05, 0.1]
train_everies = [50,100, 200]
epochs_ns = [20, 50, 100]
hidden_layers = [[11], [11,11], [20]] # [[9,9],[9,12,9]]

# понравились парам-ры:
# etas = [0.1]
# train_everies = [100]
# epochs_ns = [50]


if args.evaluate and args.filename:
    w = SimpleCarWorld(1, m, SimplePhysics, SimpleCarAgent, [9,9], 100, 20, 0.02, timedelta=0.2)
    agent = SimpleCarAgent.from_file(args.filename, [9,9])
    w.set_agents([agent])
    print(w.evaluate_agent(agent, 1200))

else:
    for e in etas:
        for t in train_everies:
            for ep in epochs_ns:
                for ls in hidden_layers:
                    for seed in seeds:
                        print("E: {0}; train: {1}; Epochs: {2}; Layers: {3}: Seed: {4}".format(e,t,ep,ls,seed))
                        np.random.seed(seed)
                        random.seed(seed)
                        m = generate_map(8, 5, 3, 3)
                        w = SimpleCarWorld(1, m, SimplePhysics, SimpleCarAgent, ls, t, ep, e, timedelta=0.2)

                        if args.filename:
                            agent = SimpleCarAgent.from_file(args.filename, ls)
                            w.set_agents([agent])

                        steps = iterations * t
                        fname = w.run(steps, visual=False)
                        new_agent = SimpleCarAgent.from_file(fname, ls)
                        # вычислить ошибку
                        # for ts in test_seeds:
                            # np.random.seed(ts)
                            # random.seed(ts)
                            # m = generate_map(8,5,3,3)
                            # w = SimpleCarWorld(1, m, SimplePhysics, SimpleCarAgent, ls, t, ep, e, timedelta=0.2)
                            # print(w.evaluate_agent(new_agent, steps, visual=False))

                        # количество кругов
                        for ts in test_seeds:
                            np.random.seed(ts)
                            random.seed(ts)
                            m = generate_map(8,5,3,3)
                            w = SimpleCarWorld(1, m, SimplePhysics, SimpleCarAgent, ls, t, ep, e, timedelta=0.2)
                            print(w.evaluate_agent(new_agent, 1200, visual=False))

                    # # if args.evaluate:
                        # # print(w.evaluate_agent(agent, steps, visual=False))
                    # # else:
                        # # w.set_agents([agent])
                        # # w.run(steps, visual=False)
