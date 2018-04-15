from os.path import join
import PreProcessing

def addToDictionary(words, dict):
    for word in words:
        if word in dict:
            dict[word] +=1
        else:
            dict[word] = 1
    return dict


def getVocabulary(dict):
    """
    Creates Vocabulary list from all the documents
    @param dict:
    @return: Dictionary of vocabulary words, key=word, value=1
    """
    vocab_dict = {}
    for k in dict.keys():
        d = dict[k]
        for word in d.keys():
            if word not in vocab_dict:
                vocab_dict[word] = 1
    # print("Vocabulary length: ", len(vocab_dict))
    return vocab_dict


def processTrainingData(class_name_list, train_file_path):
    dict = {}
    doc_dict = {}
    count = 0
    for class_name in class_name_list:
        word_count_dict = {}
        count+=1
        folder_path = join(train_file_path,class_name)
        filenames = PreProcessing.getFileNames(folder_path)
        for f in filenames:
            filepath = join(folder_path,f)
            filtered_word_list = PreProcessing.readData(filepath)
            word_count_dict = addToDictionary(filtered_word_list, word_count_dict)

        dict[count] = word_count_dict
        doc_dict[count] = len(filenames)
        print("Class: {0}, Total docs:{1}".format(class_name, doc_dict[count]))

    return (dict, doc_dict)


def getPrior(doc_dict):
    """
    Calculates the prior of each class
    @param doc_dict: Dictionary with key=class and value=number of documents
    @return: Dictionary with key=class, value=probability
    """
    prior_dict = {}
    total_docs = sum(doc_dict.values())
    for key in doc_dict.keys():
        prior_dict[key] = float(doc_dict[key]/total_docs)

    # print("prior_dict: ",prior_dict)
    return prior_dict


def getConditionalProbability(dict, vocabulary):
    """
    Calculates conditional probability of each word in the vocabulary
    @param dict:
    @param vocabulary:
    @return: Dictionary with key=(word, class), value=probability
    """
    cp_dict = {}
    vocab_count = len(vocabulary)
    for cls in dict.keys():
        word_dict = dict[cls]
        total_words_in_class = sum(word_dict.values())
        for token in vocabulary:
            t = 0
            if token in word_dict:
                t = word_dict[token]
            cp = (t + 1)/(total_words_in_class + vocab_count)
            cp_dict[(token,cls)] = cp

    # print("CP: ",len(cp_dict))
    return cp_dict


def TrainMultinomialNB(class_name_list, train_file_path):
    (dict, doc_dict) = processTrainingData(class_name_list, train_file_path)
    vocabulary_dict = getVocabulary(dict)
    prior_dict = getPrior(doc_dict)
    cp_dict = getConditionalProbability(dict, vocabulary_dict)

    return (vocabulary_dict, prior_dict, cp_dict)