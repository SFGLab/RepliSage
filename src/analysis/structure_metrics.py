"""Single-structure metrics (Rg, asphericity, volumes, ...)."""
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull

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

def ellipsoid_ratio(V):
    pca = PCA(n_components=3)
    pca.fit(V)
    axes = np.sqrt(pca.explained_variance_)
    ratio = axes.max() / axes.min()
    return ratio

def planarity(V):
    center_of_mass = np.mean(V, axis=0)
    distances = V - center_of_mass
    gyr_tensor = np.dot(distances.T, distances) / len(V)
    eigenvalues = np.linalg.eigvalsh(gyr_tensor)
    planarity_value = eigenvalues.min() / np.sum(eigenvalues)
    return planarity_value
