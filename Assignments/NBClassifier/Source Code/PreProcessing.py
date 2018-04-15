from os import listdir
from os.path import isfile, join
from stop_words import get_stop_words

stop_words = get_stop_words('english')

def getFileNames(path):
    """
    Reads all the filenames from a directory
    @param path: path to a directory
    @return: all the file names in the directory
    """
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    return onlyfiles

def normalizeWord(word):
    """
    Perform normalization of a token. For example, remove punctuations from the beginning and ending of the token
    @param word:
    @return: normalized token
    """
    word = word.lower()
    puncts = '.,?!\'\":<>();'
    word = word.strip(puncts)
    return word

def removeStopWords(words,filtered_words):
    """
    Removes stop words from a list
    @param words: list of new words
    @param filtered_words: list of already filtered words
    @return: updated filtered_words
    """
    for w in words:
        token = normalizeWord(w)
        if(token not in stop_words):
            filtered_words.append(token)
    return filtered_words


def readData(filename):
    """
    Read data from a file
    @param filename: path to the file
    @return: List of words from the file
    """
    with open(filename) as f:
        data = f.readlines()
    start = False
    emptyLine = "\n"
    filtered_word_list = []
    for line in data:
        if(start and line!=emptyLine):
            words = line.strip().split(' ')
            filtered_word_list = removeStopWords(words,filtered_word_list)
        elif (line==emptyLine):
            start = True

    # print(filtered_word_list)
    return filtered_word_list
