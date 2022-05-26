import csv
import os

import numpy as np

if __name__ == '__main__':
    path='/Users/damonx/Desktop/SDP/result/CNN+GRU/10Fold/5_16'
    # file_name_list = ["camel", "jedit", "lucene", "poi", "synapse", "xalan", "xerces"]
    f = open(path + r'/res/res_all.csv', 'w+', newline='')
    f.write('name,acc,recall,precision,F_measure,G_measure\n')
    filelist = os.listdir(path)
    print(filelist)
    for item in filelist:
        if(item.endswith('.csv')):
            with open(path+'/'+item) as csv_file:
                row = csv.reader(csv_file, delimiter=',')

                acc, recall, precision, F_measure, G_measure = [],[],[],[],[]
                for r in row:
                    print(r)
                    #直接计算
                    # acc.append(float(r[1]))
                    # recall.append(float(r[3]))
                    # precision.append(float(r[2]))
                    # F_measure.append(float(r[4]))
                    # G_measure.append(float(r[5]))

                    #手工计算
                    acc.append(float(r[0]))
                    precision.append(float(r[1]))
                    recall.append(float(r[2]))
                    F_measure.append(float(r[3]))
                    G_measure.append(float(r[4]))

                f.write(item + ',' + str(np.nanmean(acc)) + ',' + str(np.nanmean(recall)) + ',' + str(np.nanmean(precision)) + ',' + str(np.nanmean(
                    F_measure)) + ',' + str(np.nanmean(G_measure)))
                f.write('\n')

    f.close()


