import numpy as np

# returns (batch, timesteps, 4) continuous encoded labels
def labels_to_cont(labels, unique_draw):
    s = labels.shape
    labels_cont = np.zeros((s[0], s[1], 4))

    labels_cont[labels == 396, 0] = 0
    labels_cont[labels == 397, 0] = 1
    labels_cont[labels == 398, 0] = 2
    labels_cont[labels == 399, 0] = 3
    labels_cont[labels <= 90, 0] = 4
    labels_cont[((labels > 90) & (labels <= 259)), 0] = 5
    labels_cont[((labels > 259) & (labels < 396)), 0] = 6

    for i in range(s[0]):
        for j in range(s[1]):
            if labels[i][j] < 396:
                str = unique_draw[labels[i][j]]
                sep = str.split(",")
                labels_cont[i][j][1] = int(sep[0][2:])
                labels_cont[i][j][2] = int(sep[1])
                labels_cont[i][j][3] = int(sep[2][:-1])
            # else:
            #     labels_cont[i, j, 1:3] = np.random.choice(range(8, 56, 8))
            #     labels_cont[i, j, 3:] = np.random.choice(range(8, 32, 4))

    # start token
    labels_cont = np.pad(labels_cont, ((0, 0), (1, 0), (0, 0)))
    labels_cont[:, 0, 0] = 7

    return labels_cont
