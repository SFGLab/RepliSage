"""Structural metrics over a whole 3D ensemble."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from .structure_metrics import *
import os
from ..viz3d.data import list_ensemble
from ..io.structure_io import get_coordinates_cif
from ..logger import log

def compute_metrics_for_ensemble(ensemble_path, duplicated_chain=False, path=None):
    cifs = [os.path.basename(p) for _, _, p in list_ensemble(ensemble_path)]   # time order: G1 -> S -> G2/M
    Rgs, mpds, eeds, asphs, fractals, convs, gdfs, CNs = list(), list(), list(), list(), list(), list(), list(), list()
    ellipsoid_ratios, planarities, ellipsoid_volumes = list(), list(), list()
    max_pairwise_distances, bounding_box_volumes = list(), list()
    mean_distances_to_com, std_pairwise_distances = list(), list()
    max_distances_to_com = list()

    for cif in log.progress(cifs):
        V = get_coordinates_cif(ensemble_path + '/' + cif)
        if duplicated_chain:
            V = V[:len(V) // 2]
        Rgs.append(radius_of_gyration(V))
        mpds.append(mean_pairwise_distance(V))
        eeds.append(end_to_end_distance(V))
        asphs.append(asphericity(V))
        # box_counting_dimension can be time consuming
        # fractals.append(box_counting_dimension(V))
        convs.append(convex_hull_volume(V)) # can be time consuming for large point clouds
        # convs.append(convex_hull_volume(V))
        # global_distance_fluctuation can be time consuming (O(N^2) distance matrix)
        # gdfs.append(global_distance_fluctuation(V))
        CNs.append(contact_number(V))
        ellipsoid_ratios.append(ellipsoid_ratio(V))
        planarities.append(planarity(V))
        ellipsoid_volumes.append(ellipsoid_volume(V))
        max_pairwise_distances.append(max_pairwise_distance(V))
        bounding_box_volumes.append(bounding_box_volume(V))
        mean_distances_to_com.append(mean_distance_to_com(V))
        std_pairwise_distances.append(std_pairwise_distance(V))
        max_distances_to_com.append(max_distance_to_com(V))

    # Save metrics with clear names
    np.save(path + '/metadata/structural_metrics/radius_of_gyration.npy', np.array(Rgs))
    np.save(path + '/metadata/structural_metrics/mean_pairwise_distance.npy', np.array(mpds))
    np.save(path + '/metadata/structural_metrics/end_to_end_distance.npy', np.array(eeds))
    np.save(path + '/metadata/structural_metrics/asphericity.npy', np.array(asphs))
    # np.save(path + '/metadata/structural_metrics/fractal_dimension.npy', np.array(fractals))
    np.save(path + '/metadata/structural_metrics/convex_hull_volume.npy', np.array(convs))
    # np.save(path + '/metadata/structural_metrics/global_distance_fluctuation.npy', np.array(gdfs))
    np.save(path + '/metadata/structural_metrics/contact_number.npy', np.array(CNs))
    np.save(path + '/metadata/structural_metrics/ellipsoid_ratio.npy', np.array(ellipsoid_ratios))
    np.save(path + '/metadata/structural_metrics/planarity.npy', np.array(planarities))
    np.save(path + '/metadata/structural_metrics/ellipsoid_volume.npy', np.array(ellipsoid_volumes))
    np.save(path + '/metadata/structural_metrics/max_pairwise_distance.npy', np.array(max_pairwise_distances))
    np.save(path + '/metadata/structural_metrics/bounding_box_volume.npy', np.array(bounding_box_volumes))
    np.save(path + '/metadata/structural_metrics/mean_distance_to_com.npy', np.array(mean_distances_to_com))
    np.save(path + '/metadata/structural_metrics/std_pairwise_distance.npy', np.array(std_pairwise_distances))
    np.save(path + '/metadata/structural_metrics/max_distance_to_com.npy', np.array(max_distances_to_com))

    if path is not None:
        from ..plots.structure_figs import fig_structure
        fig_structure(path)
