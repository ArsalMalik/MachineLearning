import sys
import TrainMNB
import TestMNB

# class_folder_name_list = ['misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey'];
# class_folder_name_list = ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian'];
class_folder_name_list = ['talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc', 'soc.religion.christian'];
# class_folder_name_list = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x'];

def main():

    if len(sys.argv) != 3:
        print('Incorrect number of arguments.')
        print('Arg: Training folder, Test Folder')
        return

    root_folder_train = sys.argv[1]
    root_folder_test = sys.argv[2]

    print("---------  Training  --------------")
    (vocabulary_dict, prior_dict, cp_dict) = TrainMNB.TrainMultinomialNB(class_folder_name_list, root_folder_train)

    print("---------  Testing  --------------")
    (positive, negative) = TestMNB.TestMultinomialNB(class_folder_name_list, root_folder_test, vocabulary_dict, prior_dict, cp_dict)

    print("---------  Result  --------------")
    print("Postive: ", positive)
    print("Negative: ", negative)
    print("Accuracy:{0}%".format(round(positive * 100 / (positive + negative),2)))



if __name__ == '__main__':
    main()