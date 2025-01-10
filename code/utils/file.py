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


def export_results_to_file(path, results):
    with open(path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Model"])
        for r in results:
            writer.writerow([str(r)])