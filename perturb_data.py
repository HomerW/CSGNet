import numpy as np
from globals import device
from src.utils.generators.mixed_len_generator import Parser
import pickle

data_labels_paths = {3: "data/synthetic/one_op/expressions.txt",
                     5: "data/synthetic/two_ops/expressions.txt",
                     7: "data/synthetic/three_ops/expressions.txt"}
# data_labels_paths_p = {3: "data/synthetic_perturbed/one_op/expressions.txt",
#                        5: "data/synthetic_perturbed/two_ops/expressions.txt",
#                        7: "data/synthetic_perturbed/three_ops/expressions.txt"}
#
# parser = Parser()
#
# programs_p = {k: [] for k in data_labels_paths.keys()}
#
# for index in data_labels_paths.keys():
#     with open(data_labels_paths[index]) as data_file:
#         programs = data_file.readlines()
#         for i, p in enumerate(programs):
#             parsed = parser.parse(p)
#             prog_perturbed = ""
#             for token in parsed:
#                 prog_perturbed += token['value']
#                 if token['type'] == 'draw':
#                     prog_perturbed += "("
#                     params = [int(x) for x in token['param']]
#                     loc_x = params[0] + np.random.uniform(-8.0, 8.0)
#                     loc_y = params[1] + np.random.uniform(-8.0, 8.0)
#                     loc_r = params[2] + np.random.uniform(-4.0, 4.0)
#                     prog_perturbed += (str(loc_x) + ",")
#                     prog_perturbed += (str(loc_y) + ",")
#                     prog_perturbed += str(loc_r)
#                     prog_perturbed += ")"
#             programs_p[index].append(prog_perturbed)
#             print(f"{i} / {len(programs)}")
#
# for index in data_labels_paths_p.keys():
#     with open(data_labels_paths_p[index], 'w') as f:
#         for p in programs_p[index]:
#             f.write("%s\n" % p)

parser = Parser()

perturbations = {k: [] for k in data_labels_paths.keys()}

for index in data_labels_paths.keys():
    with open(data_labels_paths[index]) as data_file:
        programs = data_file.readlines()
        for i, p in enumerate(programs):
            parsed = parser.parse(p)
            perturbs = []
            for token in parsed:
                if token['type'] == 'draw':
                    loc_x = np.random.randint(-8.0, 8.0)
                    loc_y = np.random.randint(-8.0, 8.0)
                    loc_r = np.random.randint(-4.0, 4.0)
                    perturbs.append([loc_x, loc_y, loc_r])
                else:
                    perturbs.append([0.0, 0.0, 0.0])
            perturbations[index].append(perturbs)
            print(f"{i} / {len(programs)}")

with open("perturbations_more", 'wb') as f:
    f.write(pickle.dumps(perturbations))
