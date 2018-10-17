from gaitAnalysis import read_data_npy, data_segmentTS_TUG, count_TUGs, correct_mask
import numpy as np


def main():
    data = read_data_npy()
    masks, _ = data_segmentTS_TUG(data)

    patients = []
    for patient, mask in enumerate(masks, start=1):
        print('Patient %d TUGs:' % patient)
        count5 = count_TUGs(mask)
        tugs = correct_mask(count5, mask)[0]
        patient_tug = []

        for i in range(tugs[0]):
            begin = tugs[2][i]
            end = begin + tugs[1][i]
            patient_tug.append([begin, end])
            print('TUG %d: [ %d : %d ]' % (i+1, begin, end))

        patients.append(patient_tug)

    np.save('segmented_tugs.npy', patients)


if __name__ == "__main__":
    main()
