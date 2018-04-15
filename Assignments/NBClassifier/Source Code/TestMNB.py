from os.path import join
import math
import PreProcessing


def applyMultinomialNB(vocabulary_dict, prior_dict, cp_dict, word_list):
    """
    Test a document with the model
    @param vocabulary_dict: Dictionary of the vocabulary
    @param prior_dict: Dictionary of prior of each class
    @param cp_dict: Dictionary of conditional probability of all words
    @param word_list: Document's words
    @return: index of the class
    """
    score_list = []
    score = 0
    for cls in prior_dict.keys():
        score = math.log10(prior_dict[cls])
        for word in word_list:
            temp = 0
            if (word,cls) in cp_dict:
                temp = math.log10(cp_dict[(word,cls)])
            score = score + temp
        score_list.append(score)

    max_value = max(score_list)
    index = score_list.index(max_value)
    return index


def TestMultinomialNB(class_name_list, test_file_path, vocabulary_dict, prior_dict, cp_dict):
    positive = 0
    negative = 0
    for class_name in class_name_list:
        pos = 0
        neg = 0
        class_index = class_name_list.index(class_name)
        folder_path = join(test_file_path,class_name)
        doc_list = PreProcessing.getFileNames(folder_path)
        for doc in doc_list:
            filepath = join(folder_path,doc)
            filtered_word_list = PreProcessing.readData(filepath)
            index = applyMultinomialNB(vocabulary_dict, prior_dict, cp_dict, filtered_word_list)

            if (index == class_index):
                pos+=1
            else:
                neg+=1

        print("Class: {0}, Total docs:{1} => Positive:{2}, Negative:{3}".format(class_name, len(doc_list), pos, neg))
        positive = positive + pos
        negative = negative + neg

    return (positive, negative)
