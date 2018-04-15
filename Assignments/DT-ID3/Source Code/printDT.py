from Constants import NegativeClass, PositiveClass, EmptySpace

def printDecisionTree(Tree, index, msg):
    parent = Tree[index-1]

    ### Check left child
    left_child_index = 2 * index
    if(left_child_index <= len(Tree)):
        left_child = Tree[left_child_index - 1]
        ## Check if leaf node
        if(left_child==NegativeClass or left_child==PositiveClass or left_child==EmptySpace):
            print("{0}{1} = 0 : {2}".format(msg,parent,left_child))
        else:
            print("{0}{1} = 0 :".format(msg, parent))
            new_msg = msg+'|'
            printDecisionTree(Tree, left_child_index, new_msg)


    ### Check right child
    right_child_index = 2 * index + 1
    if (right_child_index <= len(Tree)):
        right_child = Tree[right_child_index - 1]
        ## Check if leaf node
        if (right_child == NegativeClass or right_child == PositiveClass or right_child==EmptySpace):
            print("{0}{1} = 1 : {2}".format(msg, parent, right_child))
        else:
            print("{0}{1} = 1 :".format(msg, parent))
            new_msg = msg + '|'
            printDecisionTree(Tree, right_child_index, new_msg)


def printTree(Tree):
    lineMsg = ''
    index = 1
    printDecisionTree(Tree, index, lineMsg)