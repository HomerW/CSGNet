import torch

def label_to_tree(label):
    stack = []
    for token in label:
        if token == 399:
            return stack[0]
        if token < 396:
            # primitive
            stack.append({"value": torch.tensor(token).long(), "left": None, "right": None})
        else:
            # operator
            obj_2 = stack.pop()
            obj_1 = stack.pop()
            stack.append({"value": torch.tensor(token).long(), "left": obj_2, "right": obj_1})
    return stack[0]
