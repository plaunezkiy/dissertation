import os
import csv


def check_or_create_file(path):
    """
    .../file.csv
    checks if exists, otherwise creates an empty file
    """
    if os.path.exists(path):
        return
    os.mknod(path)


def export_results_to_file(path, results, id_list=None):
    with open(path, "w") as f:
        writer = csv.writer(f)
        if id_list:
            writer.writerow(["", "Model"])
        else:
            writer.writerow(["Model"])
        for i, r in enumerate(results):
            if id_list:
                writer.writerow([id_list[i], str(r)])
            else:
                writer.writerow([str(r)])
