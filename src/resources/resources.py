import matplotlib
import matplotlib.pyplot as plt

file1 = "/media/sf_RECON/CA_SanDiego_2005/CA_SanDiego_las/CA_SanDiego_2005_000383.las"
file2 = "/media/sf_RECON/CA_SanDiego_2005/CA_SanDiego_las/CA_SanDiego_2005_000384.las"
file3 = "/media/sf_RECON/CA_SanDiego_2005/CA_SanDiego_las/CA_SanDiego_2005_000385.las"
file4 = "/media/sf_RECON/CA_SanDiego_2005/CA_SanDiego_las/CA_SanDiego_2005_000386.las"

files = [file1, file2, file3, file4]

oakland_tables = ["oakland_part3_ao", "oakland_part3_am", "oakland_part2_ac", "oakland_part2_al", "oakland_part3_ap","oakland_part2_ak","oakland_part2_ad","oakland_part2_ai","oakland_part2_ag","oakland_part2_aj","oakland_part2_ae","oakland_part3_ak","oakland_part2_ah","oakland_part3_an","oakland_part2_ao"]

test_pipe = {
    "pipeline": [{"type": "filters.cluster"},
                 {"type": "filters.covariancefeatures",
                  "knn": 8,
                  "threads": 2},
                 {"type": "filters.pmf"},
                 {"type": "filters.hag_nn"},
                 {"type": "filters.eigenvalues",
                  "knn": 16},
                 {"type": "filters.normal",
                  "knn": 16}
                 ]
}

## outlier settings likely need to be tuned to the dataset.
ground_pipe = {
    "pipeline": [{"type": "filters.outlier",
                  "method": "statistical",
                  "mean_k": 8,
                  "multiplier": 3.0},
                 {"type": "filters.smrf",
                  "ignore": "Classification[7:7]"},
                 {"type": "filters.range",
                  "limits": "Classification[2:2]"}
                 ]
}

def count_hash(file):
    file = open(file)
    count = 0
    for line in file:
        if line.startswith('#'):
            count += 1
    return count

def plot3d(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data['X'], data['Y'], data['Z'], c=data['label'])
    legend1 = ax.legend(*scatter.legend_elements(),
                       loc="lower left")
    ax.add_artist(legend1)
    plt.show()
