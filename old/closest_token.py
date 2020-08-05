import numpy as np

def closest_token(type, vec, unique_draw):
    distances = []
    if type == 0:
        return 396
    if type == 1:
        return 397
    if type == 2:
        return 398
    if type == 3:
        return 399
    if type == 4:
        for i in range(91):
            token_vec = []
            sep = unique_draw[i].split(",")
            token_vec.append(int(sep[0][2:]))
            token_vec.append(int(sep[1]))
            token_vec.append(int(sep[2][:-1]))
            distances.append((i, np.linalg.norm(np.array(token_vec)-vec)))
        distances.sort(key=lambda x: x[1])
        return distances[0][0]
    if type == 5:
        for i in range(91, 260):
            token_vec = []
            sep = unique_draw[i].split(",")
            token_vec.append(int(sep[0][2:]))
            token_vec.append(int(sep[1]))
            token_vec.append(int(sep[2][:-1]))
            distances.append((i, np.linalg.norm(np.array(token_vec)-vec)))
        distances.sort(key=lambda x: x[1])
        return distances[0][0]
    if type == 6:
        for i in range(260, 396):
            token_vec = []
            sep = unique_draw[i].split(",")
            token_vec.append(int(sep[0][2:]))
            token_vec.append(int(sep[1]))
            token_vec.append(int(sep[2][:-1]))
            distances.append((i, np.linalg.norm(np.array(token_vec)-vec)))
        distances.sort(key=lambda x: x[1])
        return distances[0][0]
    if type == 7:
        return 401
