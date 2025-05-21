import numpy as np
from .common import list_files_in_directory
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from tqdm import tqdm
from .common import *
from .utils import *

def radius_of_gyration(V):
    center_of_mass = np.mean(V, axis=0)
    distances = np.linalg.norm(V - center_of_mass, axis=1)
    Rg = np.sqrt(np.mean(distances**2))
    return Rg

def mean_pairwise_distance(V):
    distances = np.linalg.norm(V[:, np.newaxis] - V, axis=2)
    triu_indices = np.triu_indices(len(V), k=1)
    mean_dist = np.mean(distances[triu_indices])
    return mean_dist

def contact_number(V, cutoff=1.0):
    distances = np.linalg.norm(V[:, np.newaxis] - V, axis=2)
    np.fill_diagonal(distances, np.inf)
    contacts = np.sum(distances < cutoff, axis=1)
    mean_contacts = np.mean(contacts)
    return mean_contacts

def max_pairwise_distance(V):
    distances = np.linalg.norm(V[:, np.newaxis] - V, axis=2)
    return np.max(distances)

def bounding_box_volume(V):
    mins = np.min(V, axis=0)
    maxs = np.max(V, axis=0)
    return np.prod(maxs - mins)

def mean_distance_to_com(V):
    com = np.mean(V, axis=0)
    return np.mean(np.linalg.norm(V - com, axis=1))

def std_pairwise_distance(V):
    distances = np.linalg.norm(V[:, np.newaxis] - V, axis=2)
    triu_indices = np.triu_indices(len(V), k=1)
    return np.std(distances[triu_indices])

def max_distance_to_com(V):
    com = np.mean(V, axis=0)
    return np.max(np.linalg.norm(V - com, axis=1))

def end_to_end_distance(V):
    return np.linalg.norm(V[-1] - V[0])

def asphericity(V):
    center_of_mass = np.mean(V, axis=0)
    distances = V - center_of_mass
    gyr_tensor = np.dot(distances.T, distances) / len(V)
    eigenvalues = np.linalg.eigvalsh(gyr_tensor)
    asphericity_value = ((eigenvalues.max() - eigenvalues.min()) / 
                         np.sum(eigenvalues))
    return asphericity_value

def box_counting_dimension(V, box_sizes=np.logspace(-1, 0, 10)):
    bounds = np.ptp(V, axis=0)
    N = []
    for size in box_sizes:
        grid = np.ceil(bounds / size).astype(int)
        occupied = set(tuple((V // size).astype(int).flatten()))
        N.append(len(occupied))
    coeffs = np.polyfit(np.log(box_sizes), np.log(N), 1)
    return -coeffs[0]

def structural_anisotropy(V):
    pca = PCA(n_components=3)
    pca.fit(V)
    anisotropy = pca.explained_variance_ratio_[0] / sum(pca.explained_variance_ratio_[1:])
    return anisotropy

def ellipsoid_volume(V):
    """
    Estimate the volume of the ellipsoid defined by the principal axes of the point cloud V.
    """
    pca = PCA(n_components=3)
    pca.fit(V)
    # The axes lengths are 2*sqrt(eigenvalues) (full axes, not radii)
    axes = np.sqrt(pca.explained_variance_)
    # Volume of ellipsoid: (4/3) * pi * a * b * c, where a, b, c are the axes (radii)
    volume = (4/3) * np.pi * axes[0] * axes[1] * axes[2]
    return volume

def convex_hull_volume(V):
    hull = ConvexHull(V)
    hull_volume = hull.volume
    return hull_volume

def global_distance_fluctuation(V):
    distances = np.linalg.norm(V[:, np.newaxis] - V, axis=2)
    gdf = np.std(distances)
    return gdf

# New metric: Ellipsoid ratio (largest/smallest principal axis)
def ellipsoid_ratio(V):
    pca = PCA(n_components=3)
    pca.fit(V)
    axes = np.sqrt(pca.explained_variance_)
    ratio = axes.max() / axes.min()
    return ratio

# New metric: Planarity (smallest eigenvalue / sum of eigenvalues of gyration tensor)
def planarity(V):
    center_of_mass = np.mean(V, axis=0)
    distances = V - center_of_mass
    gyr_tensor = np.dot(distances.T, distances) / len(V)
    eigenvalues = np.linalg.eigvalsh(gyr_tensor)
    planarity_value = eigenvalues.min() / np.sum(eigenvalues)
    return planarity_value

def compute_metrics_for_ensemble(ensemble_path, duplicated_chain=False, path=None):
    cifs = list_files_in_directory(ensemble_path)
    Rgs, mpds, eeds, asphs, fractals, convs, gdfs, CNs = list(), list(), list(), list(), list(), list(), list(), list()
    ellipsoid_ratios, planarities, ellipsoid_volumes = list(), list(), list()
    max_pairwise_distances, bounding_box_volumes = list(), list()
    mean_distances_to_com, std_pairwise_distances = list(), list()
    max_distances_to_com = list()

    for cif in tqdm(cifs):
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

    # Plotting
    figure(figsize=(10, 6), dpi=400)
    plt.plot(Rgs, 'r-')
    plt.xlabel('sample number', fontsize=16)
    plt.ylabel('Radius of Gyration', fontsize=16)
    if path is not None:
        plt.savefig(path + '/plots/structural_metrics/radius_of_gyration.svg', format='svg', dpi=400)
    plt.close()

    figure(figsize=(10, 6), dpi=400)
    plt.plot(mpds, 'k-')
    plt.xlabel('sample number', fontsize=16)
    plt.ylabel('Mean Pairwise Distance', fontsize=16)
    if path is not None:
        plt.savefig(path + '/plots/structural_metrics/mean_pairwise_distance.svg', format='svg', dpi=400)
    plt.close()

    figure(figsize=(10, 6), dpi=400)
    plt.plot(eeds, 'k-')
    plt.xlabel('sample number', fontsize=16)
    plt.ylabel('End to End Distance', fontsize=16)
    if path is not None:
        plt.savefig(path + '/plots/structural_metrics/end_to_end_distance.svg', format='svg', dpi=400)
    plt.close()

    figure(figsize=(10, 6), dpi=400)
    plt.plot(asphs, 'k-')
    plt.xlabel('sample number', fontsize=16)
    plt.ylabel('Asphericity', fontsize=16)
    if path is not None:
        plt.savefig(path + '/plots/structural_metrics/asphericity.svg', format='svg', dpi=400)
    plt.close()

    # figure(figsize=(10, 6), dpi=400)
    # plt.plot(fractals, 'k-')
    # plt.xlabel('sample number', fontsize=16)
    # plt.ylabel('Fractal Dimension', fontsize=16)
    # if path is not None:
    #     plt.savefig(path + '/plots/structural_metrics/fractal_dimension.svg', format='svg', dpi=400)
    # plt.close()

    figure(figsize=(10, 6), dpi=400)
    plt.plot(convs, 'k-')
    plt.xlabel('sample number', fontsize=16)
    plt.ylabel('Convex Hull Volume', fontsize=16)
    if path is not None:
        plt.savefig(path + '/plots/structural_metrics/convex_hull_volume.svg', format='svg', dpi=400)
    plt.close()

    # figure(figsize=(10, 6), dpi=400)
    # plt.plot(gdfs, 'k-')
    # plt.xlabel('sample number', fontsize=16)
    # plt.ylabel('Global Distance Fluctuation', fontsize=16)
    # if path is not None:
    #     plt.savefig(path + '/plots/structural_metrics/global_distance_fluctuation.svg', format='svg', dpi=400)
    # plt.close()

    figure(figsize=(10, 6), dpi=400)
    plt.plot(CNs, 'k-')
    plt.xlabel('sample number', fontsize=16)
    plt.ylabel('Contact Number', fontsize=16)
    if path is not None:
        plt.savefig(path + '/plots/structural_metrics/contact_number.svg', format='svg', dpi=400)
    plt.close()

    figure(figsize=(10, 6), dpi=400)
    plt.plot(ellipsoid_ratios, 'b-')
    plt.xlabel('sample number', fontsize=16)
    plt.ylabel('Ellipsoid Ratio', fontsize=16)
    if path is not None:
        plt.savefig(path + '/plots/structural_metrics/ellipsoid_ratio.svg', format='svg', dpi=400)
    plt.close()

    figure(figsize=(10, 6), dpi=400)
    plt.plot(planarities, 'g-')
    plt.xlabel('sample number', fontsize=16)
    plt.ylabel('Planarity', fontsize=16)
    if path is not None:
        plt.savefig(path + '/plots/structural_metrics/planarity.svg', format='svg', dpi=400)
    plt.close()

    figure(figsize=(10, 6), dpi=400)
    plt.plot(ellipsoid_volumes, 'navy')
    plt.xlabel('sample number', fontsize=16)
    plt.ylabel('Ellipsoid Volume', fontsize=16)
    if path is not None:
        plt.savefig(path + '/plots/structural_metrics/ellipsoid_volume.svg', format='svg', dpi=400)
    plt.close()

    figure(figsize=(10, 6), dpi=400)
    plt.plot(max_pairwise_distances, 'm-')
    plt.xlabel('sample number', fontsize=16)
    plt.ylabel('Max Pairwise Distance', fontsize=16)
    if path is not None:
        plt.savefig(path + '/plots/structural_metrics/max_pairwise_distance.svg', format='svg', dpi=400)
    plt.close()

    figure(figsize=(10, 6), dpi=400)
    plt.plot(bounding_box_volumes, 'c-')
    plt.xlabel('sample number', fontsize=16)
    plt.ylabel('Bounding Box Volume', fontsize=16)
    if path is not None:
        plt.savefig(path + '/plots/structural_metrics/bounding_box_volume.svg', format='svg', dpi=400)
    plt.close()

    figure(figsize=(10, 6), dpi=400)
    plt.plot(mean_distances_to_com, 'y-')
    plt.xlabel('sample number', fontsize=16)
    plt.ylabel('Mean Distance to COM', fontsize=16)
    if path is not None:
        plt.savefig(path + '/plots/structural_metrics/mean_distance_to_com.svg', format='svg', dpi=400)
    plt.close()

    figure(figsize=(10, 6), dpi=400)
    plt.plot(std_pairwise_distances, color='orange')
    plt.xlabel('sample number', fontsize=16)
    plt.ylabel('Std Pairwise Distance', fontsize=16)
    if path is not None:
        plt.savefig(path + '/plots/structural_metrics/std_pairwise_distance.svg', format='svg', dpi=400)
    plt.close()

    figure(figsize=(10, 6), dpi=400)
    plt.plot(max_distances_to_com, color='purple')
    plt.xlabel('sample number', fontsize=16)
    plt.ylabel('Max Distance to COM', fontsize=16)
    if path is not None:
        plt.savefig(path + '/plots/structural_metrics/max_distance_to_com.svg', format='svg', dpi=400)
    plt.close()