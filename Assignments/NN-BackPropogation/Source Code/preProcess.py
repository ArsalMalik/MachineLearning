import sys
import csv
import statistics
from os import path


def removeRows(filename):
    with open(filename, 'rt') as f:
        reader = csv.reader(f, delimiter=',')
        #data =list(reader)
        #print("The number of rows in original dataset: ", len(data))
        UpdatedDataset = []
        for row in reader:
            if len(row):
                if checkRow(row) == "Yes":
                    continue
                else:
                    UpdatedDataset.append(row)
        return UpdatedDataset


        # Check for null, empty or missing values


def checkRow(row):
    rowData = row
    for i in range(len(rowData)):
        if rowData[i] == ' ?' or rowData[i] == '' or rowData[i] is None:
            return "Yes"


# removeMissingRows(filename)

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def isNumeric(data, col):
    for row in range(len(data)):
        if isfloat(data[row][col].strip()):
            continue
        else:
            return False
    return True

def standardization(UpdatedDataset):
    total_records = len(UpdatedDataset)
    no_of_features = len(UpdatedDataset[0])
    col = []
    print("Total records:{0} and Total Attributes:{1}".format(total_records,no_of_features));
    #print(isfloat(UpdatedDataset[0][2].strip()))

    for j in range(no_of_features):
        if isNumeric(UpdatedDataset, j):
            for i in range(total_records):
                col.append(float(UpdatedDataset[i][j]))

            Mean = statistics.mean(col)
            print("Mean:", Mean)
            stDev = statistics.stdev(col)
            print("stDev:", stDev)
            for i in range(total_records):
                # Standardization formula
                UpdatedDataset[i][j] = round((col[i] - Mean) / stDev, 2)
        else:
            attr_name = []
            for i in range(total_records):
                if(UpdatedDataset[i][j] in attr_name):
                    index = attr_name.index(UpdatedDataset[i][j])
                    UpdatedDataset[i][j] = index+1
                else:
                    attr_name.append(UpdatedDataset[i][j])
                    UpdatedDataset[i][j] = len(attr_name)

    return UpdatedDataset

def writeFile(data, outputFilename):
    total_records = len(data)
    no_of_features = len(data[0])
    text_file = open(outputFilename, "w")
    for i in range(total_records):
        line = ""
        for j in range(no_of_features):
            line = line + str(data[i][j])
            if j != no_of_features-1:
                line = line+","
        line = line + "\n"
        text_file.write(line);
    text_file.close()


def preProcess(filepath, outputFilename):
    dataset = removeRows(filepath)
    print("Number of records after removing records with missing elements: ", len(dataset))
    standardizedDataset = standardization(dataset)
    print("Dataset after applying standardization is written to file: ", outputFilename)

    writeFile(standardizedDataset, outputFilename)


def main():
    if len(sys.argv) < 3:
        print('Incorrect number of arguments.')
        print('Arg: input dataset, output dataset')
        return

    filename = sys.argv[1]
    outputFilename = sys.argv[2]

    if ( path.exists(filename) == False):
        print("Error! Wrong path to file!")
        return

    preProcess(filename, outputFilename)


if __name__ == '__main__':
    main()