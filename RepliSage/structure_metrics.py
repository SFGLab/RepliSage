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
    # Take only upper triangle distances (no duplicates, no diagonal)
    triu_indices = np.triu_indices(len(V), k=1)
    mean_dist = np.mean(distances[triu_indices])
    return mean_dist

def contact_number(V, cutoff=1.0):
    distances = np.linalg.norm(V[:, np.newaxis] - V, axis=2)
    # Ignore self-contacts by setting diagonal to a large value
    np.fill_diagonal(distances, np.inf)
    contacts = np.sum(distances < cutoff, axis=1)
    mean_contacts = np.mean(contacts)
    return mean_contacts

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
    bounds = np.ptp(V, axis=0)  # extent of the structure
    N = []
    
    for size in box_sizes:
        grid = np.ceil(bounds / size).astype(int)  # number of boxes in each dimension
        occupied = set(tuple((V // size).astype(int).flatten()))  # check occupied boxes
        N.append(len(occupied))
    
    # Log-log fit for slope as fractal dimension
    coeffs = np.polyfit(np.log(box_sizes), np.log(N), 1)
    return -coeffs[0]

def structural_anisotropy(V):
    pca = PCA(n_components=3)
    pca.fit(V)
    # Ratio of explained variance by the first component to the sum of others
    anisotropy = pca.explained_variance_ratio_[0] / sum(pca.explained_variance_ratio_[1:])
    return anisotropy

def convex_hull_volume(V):
    hull = ConvexHull(V)
    hull_volume = hull.volume
    return hull_volume

def global_distance_fluctuation(V):
    distances = np.linalg.norm(V[:, np.newaxis] - V, axis=2)
    gdf = np.std(distances)
    return gdf

def compute_metrics_for_ensemble(ensemble_path,duplicated_chain=False,path=None):
    cifs = list_files_in_directory(ensemble_path)
    Rgs, mpds, eeds, asphs, fractals, convs, gdfs, CNs = list(), list(), list(), list(), list(), list(), list(), list()

    for cif in tqdm(cifs):
        V = get_coordinates_cif(ensemble_path+'/'+cif)
        if duplicated_chain: V = V[:len(V)//2]
        Rgs.append(radius_of_gyration(V))
        mpds.append(mean_pairwise_distance(V))
        eeds.append(end_to_end_distance(V))
        asphs.append(asphericity(V))
        fractals.append(box_counting_dimension(V))
        convs.append(convex_hull_volume(V))
        gdfs.append(global_distance_fluctuation(V))
        CNs.append(contact_number(V))
    np.save(path+'/other/Rgs.npy',np.array(Rgs))
    np.save(path+'/other/mpds.npy',np.array(mpds))
    np.save(path+'/other/eeds.npy',np.array(eeds))
    np.save(path+'/other/asphs.npy',np.array(asphs))
    np.save(path+'/other/fractal_dims.npy',np.array(fractals))
    np.save(path+'/other/convex_hull_volume.npy',np.array(convs))
    np.save(path+'/other/gdfs.npy',np.array(gdfs))
    np.save(path+'/other/CNs.npy',np.array(CNs))


    figure(figsize=(10, 6), dpi=400)
    plt.plot(Rgs,'r-')
    plt.xlabel('sample number', fontsize=16)
    plt.ylabel('Gyration Radius', fontsize=16)
    if path!=None:
        plt.savefig(path+'/plots/Rg.svg',format='svg',dpi=400)
    plt.close()

    figure(figsize=(10, 6), dpi=400)
    plt.plot(mpds,'k-')
    plt.xlabel('sample number', fontsize=16)
    plt.ylabel('Mean Pairwise Distance', fontsize=16)
    if path!=None:
        plt.savefig(path+'/plots/pairwise.svg',format='svg',dpi=400)
    plt.close()
    
    figure(figsize=(10, 6), dpi=400)
    plt.plot(eeds,'k-')
    plt.xlabel('sample number', fontsize=16)
    plt.ylabel('End to End Distance', fontsize=16)
    if path!=None:
        plt.savefig(path+'/plots/end2end.svg',format='svg',dpi=400)
    plt.close()

    figure(figsize=(10, 6), dpi=400)
    plt.plot(asphs,'k-')
    plt.xlabel('sample number', fontsize=16)
    plt.ylabel('Asphericity', fontsize=16)
    if path!=None:
        plt.savefig(path+'/plots/asphericity.svg',format='svg',dpi=400)
    plt.close()

    figure(figsize=(10, 6), dpi=400)
    plt.plot(fractals,'k-')
    plt.xlabel('sample number', fontsize=16)
    plt.ylabel('Fractal Dimension', fontsize=16)
    if path!=None:
        plt.savefig(path+'/plots/fractal.svg',format='svg',dpi=400)
    plt.close()

    figure(figsize=(10, 6), dpi=400)
    plt.plot(convs,'k-')
    plt.xlabel('sample number', fontsize=16)
    plt.ylabel('Convex Volume', fontsize=16)
    if path!=None:
        plt.savefig(path+'/plots/convex_vol.svg',format='svg',dpi=400)
    plt.close()

    figure(figsize=(10, 6), dpi=400)
    plt.plot(gdfs,'k-')
    plt.xlabel('sample number', fontsize=16)
    plt.ylabel('Global Distance Fluctuations', fontsize=16)
    if path!=None:
        plt.savefig(path+'/plots/global_dist_flucts.svg',format='svg',dpi=400)
    plt.close()

    figure(figsize=(10, 6), dpi=400)
    plt.plot(CNs,'k-')
    plt.xlabel('sample number', fontsize=16)
    plt.ylabel('Contact Number', fontsize=16)
    if path!=None:
        plt.savefig(path+'/plots/contact_number.svg',format='svg',dpi=400)
    plt.close()
