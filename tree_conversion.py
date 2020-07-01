import torch

device = torch.device("cuda")

def label_to_tree(label):
    stack = []
    for token in label:
        if token == 399:
            return stack[0]
        if token < 396:
            # primitive
            stack.append({"value": torch.tensor(token).long().to(device), "left": None, "right": None})
        else:
            # operator
            obj_2 = stack.pop()
            obj_1 = stack.pop()
            stack.append({"value": torch.tensor(token).long().to(device), "left": obj_2, "right": obj_1})
    return stack[0]

def tree_to_label(tree):
    def flatten_recur(node):
        if node["right"] is None and node["left"] is None:
            return [node["value"]]
        else:
            lchild = flatten_recur(node["left"])
            rchild = flatten_recur(node["right"])
            return [node["value"]] + lchild + rchild
    return torch.stack(list(reversed(flatten_recur(tree))))
