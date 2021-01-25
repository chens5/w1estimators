import gudhi.hera
import gudhi.wasserstein.barycenter
import gudhi.wasserstein
import csv
import numpy as np
import pd_estimators as pde
import os
import time
import random
import sys
import faulthandler
import math

# Converts a csv file to a list of points
# Also returns the representation of the diagram required to calculate
# flowtree/embedding distance (i.e. [(a1, 1.0),... ]) where a1 is the index of
# the point in the total list of points.
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
    if len(sys.argv) < 2:
        sys.exit("Need data folder")
    folder = sys.argv[1]
    data = load_data(folder)
    flowtree_vocabulary = data[0].astype(np.float32)
    diagrams = data[1]
    flowtree_diagrams = data[2]

    num_diagrams = len(diagrams)

    queries = random.sample(range(200), 100)
    candidates = [i for i in range(num_diagrams) if i not in queries]

    solver = pde.PDEstimators()
    start = time.time()
    solver.load_points(flowtree_vocabulary)
    end = time.time()
    print("Points in flowtree: ", len(flowtree_vocabulary))
    print("Time for building the flowtree: ", end - start)
    start = time.time()
    solver.load_diagrams(flowtree_diagrams)
    end = time.time()
    print("Time for loading all embedding representations: ", end - start)

    ft_results = []
    emb_results = []
    hera_results = []
    for i in queries:
        for j in candidates:
            hera_results.append(gudhi.hera.wasserstein_distance(diagrams[i], diagrams[j], order =1, internal_p=2))
            ft_results.append(solver.flowtree_distance(flowtree_diagrams[i], flowtree_diagrams[j], 2))
            emb_results.append(solver.embedding_distance(i, j))

    ft_results = np.array(ft_results)
    emb_results = np.array(emb_results)
    hera_results = np.array(hera_results)
    print('Average flowtree error:', np.average(abs(ft_results - hera_results)/hera_results))
    print('Average embedding error:', np.average(abs(emb_results - hera_results)/hera_results))





if __name__ == '__main__':
    faulthandler.enable()
    main()
