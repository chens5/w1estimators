import gudhi.hera
import gudhi.wasserstein.barycenter
import gudhi.wasserstein
import csv
import numpy as np
import matplotlib.pyplot as plt
import pd_estimators as pde
import os
import time
import random
import sys
import faulthandler
import math

# process p diagrams into numpy arrays
def csv_to_diagram(file, dict, unique_points):
    p_list = []
    ft_diagram = []
    with open(file) as f:
        reader = csv.reader(f)
        for row in reader:
            birth = float(row[0])
            death = float(row[1])
            p = (birth, death)
            p_list.append((birth, death))
            if p not in dict:
                dict[p] = len(unique_points)
                unique_points.append(p)
            ft_diagram.append((dict[p], 1.0))
    f.close()
    diagram = np.asarray(p_list)
    return p_list, ft_diagram


def load_data(folder):
    basepath = folder
    all_files = []
    for entry in os.listdir(basepath):
        if os.path.isfile(os.path.join(basepath, entry)):
            all_files.append(os.path.join(basepath, entry))
    diagrams = []
    ft_diagrams = []
    unique_pts = []
    dict_pt = {}
    for file in all_files:
        diagram, ft_diagram = csv_to_diagram(file, dict_pt, unique_pts)
        diagrams.append(diagram)
        ft_diagrams.append(ft_diagram)
    vocab = np.array(unique_pts)
    return vocab, diagrams, ft_diagrams


def main():
    if len(sys.argv) < 3:
        sys.exit("Need data folder and input file to write results")
    file = sys.argv[2]
    folder = sys.argv[1]
    data = load_data(folder)
    flowtree_vocabulary = data[0].astype(np.float32)
    all_diagrams = data[1]
    flowtree_diagrams = data[2]

    num_diagrams = len(all_diagrams)
    results = []
    queries = random.sample(range(200), 100)
    candidates = [i for i in range(num_diagrams) if i not in queries]


    #solver.quadtree_query_pair(103, 11)
    with open(file, mode='w') as csv_file:
        fieldnames = ['query', 'candidate', 'max_diagram_size', "hera_time", "ft_time", "qt_time", "hera_res", "ft_res", "qt_res"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for trial in range(5):
            solver = pde.PDEstimators()
            start = time.time()
            solver.load_vocabulary(flowtree_vocabulary)
            end = time.time()
            print("Points in flowtree: ", len(flowtree_vocabulary))
            print("Time for building the flowtree: ", end - start)


            start = time.time()
            solver.load_diagrams(flowtree_diagrams)
            end = time.time()
            print("Time for computing all embeddings: ", end - start)



if __name__ == '__main__':
    faulthandler.enable()
    main()
