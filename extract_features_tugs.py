from gaitAnalysis import *

data = read_data_npy()
matrix = data_features(data)
masks, segm = data_segmentTS_TUG(data)

patients = []
for mask in masks:
    count5 = count_TUGs(mask)
    tugs = correct_mask(count5, mask)[0]
    patient_tug = []

    for i in range(tugs[0]):
        print(i)
        begin = tugs[2][i]
        end = begin + tugs[1][i]
        patient_tug.append([begin, end])
        print('begin:\t' + str(begin))
        print('end:\t' + str(end))

    patients.append(patient_tug)

np.save('segmented_tugs.npy', patients)
