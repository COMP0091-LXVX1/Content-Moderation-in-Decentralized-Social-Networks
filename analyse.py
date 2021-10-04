import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DATASET_SET = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
NUM_ENTRIES_SET = [10000, 50000, 200000]

f2 = np.zeros((len(NUM_ENTRIES_SET), len(DATASET_SET)), dtype=float)
f3 = np.zeros((len(NUM_ENTRIES_SET), len(DATASET_SET)), dtype=float)

set = "TEST"

for a, DATASET in enumerate(DATASET_SET):
    for b, NUM_ENTRIES in enumerate(NUM_ENTRIES_SET):
        print("FOR DATASET: " + str(DATASET) + "; NUM_ENTRIES: " + str(NUM_ENTRIES))
        print()

        if set == "VAL":
            output = np.loadtxt("output_val-" + str(DATASET) + "-" + str(NUM_ENTRIES) + ".npy")
            yval = np.loadtxt("y_val-" + str(DATASET) + "-" + str(NUM_ENTRIES) + ".npy")
        if set == "TEST":
            output = np.loadtxt("output_test-" + str(DATASET) + "-" + str(NUM_ENTRIES) + ".npy")
            yval = np.loadtxt("y_test-" + str(DATASET) + "-" + str(NUM_ENTRIES) + ".npy")

        x = yval/output

        print("F=2: ", np.sum(np.logical_and(1/2 < x, x < 2)) / len(x))
        f2[b][a] = np.sum(np.logical_and(1/2 < x, x < 2)) / len(x)

        print("F=3: ", np.sum(np.logical_and(1/3 < x, x < 3)) / len(x))
        f3[b][a] = np.sum(np.logical_and(1/3 < x, x < 3)) / len(x)

        fig, ax = plt.subplots()

        MAX_INT = 5
        x = x[x < MAX_INT]
        NO_BINS = 50
        N, bins, patches = ax.hist(x, bins=NO_BINS, edgecolor='white', linewidth=1)

        for i in range(1, NO_BINS+1):
            if 1/2 < i * MAX_INT / NO_BINS <= 2:
                patches[i-1].set_facecolor('green')
            else:
                if 1/3 < i * MAX_INT / NO_BINS <= 3:
                    patches[i-1].set_facecolor('skyblue')
                else:
                    patches[i-1].set_facecolor('tomato')

        ax.set_xlabel("ratio between predict spreading and true spreading for " + str(NUM_ENTRIES) + " data points")
        ax.set_ylabel("number of entries")

        red_patch = mpatches.Patch(color='tomato', label='outside intervals')
        blue_patch = mpatches.Patch(color='skyblue', label='between [1/3, 3]')
        green_patch = mpatches.Patch(color='green', label='between [1/2, 2]')
        plt.legend(handles=[green_patch, blue_patch, red_patch])

        plt.savefig("fraction-" + str(DATASET) + "-" + str(NUM_ENTRIES) + "-" + set +".png")

        print()
        print()

for j, NUM_ENTRIES in enumerate(NUM_ENTRIES_SET):
    print(NUM_ENTRIES, ":", np.mean(f2[j]), np.std(f2[j]), " | ", np.mean(f3[j]), np.std(f3[j]))
