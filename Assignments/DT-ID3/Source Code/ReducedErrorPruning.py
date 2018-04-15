import pandas as pd
import math
import random
from Constants import NegativeClass, PositiveClass, EmptySpace, TargetLabel
from collections import OrderedDict


def remove_empty_spaces_from_tree(t):
    # Delete empty spaces from the end of the tree
    for element in reversed(t):
        if element == EmptySpace:
            t.pop()
        else:
            break
    return t

def pruneNode(t, index):
    left_child_index = 2 * index
    right_child_index = 2 * index + 1

    if left_child_index <= len(t):
        t[left_child_index - 1] = EmptySpace
        pruneNode(t, left_child_index)

    if right_child_index <= len(t):
        t[right_child_index-1] = EmptySpace
        pruneNode(t, right_child_index)

    return t



def selectMajorityClass(data, t, index):
    i = index
    ancestors = {}
    while True:
        value = 1
        if(i%2==0):
            value = 0
        i = math.floor(i/2)
        ancestors[i] = value
        if(i==1):
            break

    od = OrderedDict(sorted(ancestors.items()))
    datas = data
    for k,v in od.items():
        #print(k,v)
        datas = datas.where(datas[t[k-1]] == v).dropna()

    n = datas.where(datas[TargetLabel] == NegativeClass).dropna().shape[0]
    p = datas.shape[0] - n

    if n > p:
        return NegativeClass
    else:
        return PositiveClass


def get_random_index(t, further_nodes_to_remove):
    indexes = []
    count = 1
    for i in t:
        ### only taking into account the internal nodes
        if i != EmptySpace and i != PositiveClass and i != NegativeClass and count!=1:
            indexes.append(count)
        count+=1

    #check if enough internal nodes are available
    if(further_nodes_to_remove > len(indexes)):
        return (0, False)
    else:
        random_index = random.choice(indexes)
        return (random_index, True)


def pruneTree(data, t, pruning_factor):
    total_nodes_of_tree = len(t) - t.count(EmptySpace) - t.count(PositiveClass) - t.count(NegativeClass)
    total_nodes_to_prune = math.floor(pruning_factor * total_nodes_of_tree)
    #print(pruning_factor,total_nodes_of_tree)
    #print("Nodes to be removed:{0}".format(total_nodes_to_prune))
    # counter for the number of nodes removed
    nodes_removed = 0
    pruned_tree = t

    while nodes_removed < total_nodes_to_prune:
        # getting a random index
        pruned_tree = remove_empty_spaces_from_tree(pruned_tree)
        random = get_random_index(pruned_tree, total_nodes_to_prune-nodes_removed)
        if random[1] == True:
            random_index = random[0]
        else:
            #print("Not enough nodes to prune")
            break

        if pruned_tree[random_index-1] == EmptySpace or pruned_tree[random_index-1] == PositiveClass or pruned_tree[random_index-1] == NegativeClass:
            #print("Can not prune leaf nodes!")
            continue
        elif random_index == 1:
            print("Can not prune Root node!")
            continue
        else:
            pruned_tree = pruneNode(pruned_tree, random_index)
            ### calculate majority class for this node
            majority_class = selectMajorityClass(data, pruned_tree, random_index)
            pruned_tree[random_index-1] = majority_class
            nodes_removed += 1
            #print("##### Random index: {0} - {1} Nodes removed out of {2}".format(random_index,nodes_removed,total_nodes_to_prune))


    #print("Pruned Tree:{0}".format(pruned_tree))
    return pruned_tree