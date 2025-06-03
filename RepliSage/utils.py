#########################################################################
########### CREATOR: SEBASTIAN KORSAK, WARSAW 2022 ######################
#########################################################################

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for thread safety
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import pandas as pd
from scipy.spatial import distance
from importlib.resources import files
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

def binding_vectors_from_bedpe(bedpe_file,N_beads,region,chrom,normalization=False,viz=False):
    '''
    Definition of left and right CTCF binding potential.

    Input:
    bedpe_file (str): path with bepde file with loops
    region (list): a list with two integers [start,end], which represent the start and end point of the region of interest.
    chrom (str): chromosome of interest.
    normalization (bool): in case that it is needed to normalize to numpy arrays that represent left and right CTCF binding potential.
    viz (bool): If True is vizualizes the distribution of distances of loops from the diagonal and the binding potentials as functions of simulated polymer distance.

    Output:
    L (numpy array): left CTCF binding potential.
    R (numpy array): right CTCF binding potential.
    dists (numpy array): distances of CTCF loops from the diagonal.
    '''
    # Read file and select the region of interest
    df = pd.read_csv(bedpe_file,sep='\t',header=None)
    df = df[(df[1]>=region[0])&(df[2]>=region[0])&(df[4]>=region[0])&(df[5]>=region[0])&(df[5]<region[1])&(df[4]<region[1])&(df[1]<region[1])&(df[2]<region[1])&(df[0]==chrom)].reset_index(drop=True)
    if df.empty:
        raise ValueError("\033[91mThe selected region or chromosome is invalid: the final set does not contain any loops.\033[0m")

    # Convert hic coords into simulation beads
    resolution = (region[1]-region[0])//N_beads
    df[1], df[2], df[4], df[5] = (df[1]-region[0])//resolution, (df[2]-region[0])//resolution, (df[4]-region[0])//resolution, (df[5]-region[0])//resolution
    
    # Check if columns 7 and 8 exist
    has_col_7_8 = df.shape[1] > 8
    if has_col_7_8:
        print("\033[92mThe input bedpe file contains CTCF orientation! It will run taking it into account.\033[0m")
    else:
        print("\033[93mWARNING: The input bedpe file does not contain CTCF orientation and thus it is not taken into account. If you would like to add CTCF orientation check the documentation.\033[0m")

    # Compute the matrix
    distances = list()
    J = np.zeros((N_beads, N_beads), dtype=np.float64)
    L, R = np.zeros(N_beads, dtype=np.float64), np.zeros(N_beads, dtype=np.float64)
    for i in range(len(df)):
        x, y = min((df[1][i] + df[2][i]) // 2, N_beads - 1), min((df[4][i] + df[5][i]) // 2, N_beads - 1)
        distances.append(distance_point_line(x, y))
        J[x, y] = 0
        J[y, x] = 0
        strength = df[6][i]
        # If strength is None or negative, skip adding signal
        if pd.isna(strength) or strength is None or strength < 0:
            continue
        if has_col_7_8:
            if df[7][i] >= 0:
                L[x] += strength * (1 - df[7][i])
            if df[8][i] >= 0:
                L[y] += strength * (1 - df[8][i])
            if df[7][i] >= 0:
                R[x] += strength * df[7][i]
            if df[8][i] >= 0:
                R[y] += strength * df[8][i]
        else:
            L[x] += strength
            L[y] += strength
            R[x] += strength
            R[y] += strength
    
    # Normalize (if neccesary): it means to convert values to probabilities
    if normalization:
        L, R = L/np.sum(L), R/np.sum(R)

    if viz:
        fig, axs = plt.subplots(2, figsize=(15, 10))
        axs[0].plot(L,'g-')
        axs[0].set_ylabel('Left potential',fontsize=16)
        axs[1].plot(R,'r-')
        axs[1].set_ylabel('Right potential',fontsize=16)
        axs[1].set_xlabel('Genomic Distance (with simumation beads as a unit)',fontsize=16)
        fig.show()

    return L, R, J

def distance_point_line(x0,y0,a=1,b=-1,c=0):
    return np.abs(a*x0+b*y0+c)/np.sqrt(a**2+b**2)

def make_folder(folder_name):
    subfolders = [
        'plots',
        'metadata',
        'ensemble',
        'plots/MCMC_diagnostics', 
        'plots/structural_metrics', 
        'plots/graph_metrics',
        'plots/md_simulation',
        'plots/replication_simulation', 
        'metadata/energy_factors',
        'metadata/MCMC_output',
        'metadata/structural_metrics', 
        'metadata/graph_metrics',
        'metadata/md_dynamics',
    ]
    created_any = False
    for subfolder in subfolders:
        path = os.path.join(folder_name, subfolder)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            created_any = True
    if created_any:
        print(f'\033[92mDirectories were created in "{folder_name}".\033[0m')
    else:
        print(f'\033[94mAll necessary folders already exist in "{folder_name}".\033[0m')
    return folder_name

############# Creation of mmcif and psf files #############
mmcif_atomhead = """data_nucsim
# 
_entry.id nucsim
# 
_audit_conform.dict_name       mmcif_pdbx.dic 
_audit_conform.dict_version    5.296 
_audit_conform.dict_location   http://mmcif.pdb.org/dictionaries/ascii/mmcif_pdbx.dic 
# ----------- ATOMS ----------------
loop_
_atom_site.group_PDB 
_atom_site.id 
_atom_site.type_symbol 
_atom_site.label_atom_id 
_atom_site.label_alt_id 
_atom_site.label_comp_id 
_atom_site.label_asym_id 
_atom_site.label_entity_id 
_atom_site.label_seq_id 
_atom_site.pdbx_PDB_ins_code 
_atom_site.Cartn_x 
_atom_site.Cartn_y 
_atom_site.Cartn_z
"""

mmcif_connecthead = """#
loop_
_struct_conn.id
_struct_conn.conn_type_id
_struct_conn.ptnr1_label_comp_id
_struct_conn.ptnr1_label_asym_id
_struct_conn.ptnr1_label_seq_id
_struct_conn.ptnr1_label_atom_id
_struct_conn.ptnr2_label_comp_id
_struct_conn.ptnr2_label_asym_id
_struct_conn.ptnr2_label_seq_id
_struct_conn.ptnr2_label_atom_id
"""

def write_cmm(comps,name):
    comp_old = 2
    counter, start = 0, 0
    comp_dict = {-1:'red', 1:'blue'}
    content = ''

    for i, comp in enumerate(comps):
        if comp_old==comp:
            counter+=1
        elif i!=0:
            content+=f'color {comp_dict[comp_old]} :{start}-{start+counter+1}\n'
            counter, start = 0, i
        comp_old=comp

    content+=f'color {comp_dict[comp]} :{start}-{start+counter+1}\n'
    with open(name, 'w') as f:
        f.write(content)

def write_mmcif(points1,points2=None,cif_file_name='LE_init_struct.cif'):
    atoms = ''
    run_repli = np.all(points2!=None)
    n = len(points1)
    for i in range(0,n):
        x = points1[i][0]
        y = points1[i][1]
        try:
            z = points1[i][2]
        except IndexError:
            z = 0.0
        atom_type =  'ALB' if i==0 or i==n-1 else 'ALA'  
        atoms += ('{0:} {1:} {2:} {3:} {4:} {5:} {6:}  {7:} {8:} '
                '{9:} {10:.3f} {11:.3f} {12:.3f}\n'.format('ATOM', i+1, 'D', 'CA',\
                                                            '.', 'ALA', 'A', 1, i+1, '?',\
                                                            x, y, z))
    
    if run_repli:
        for i in range(0,n):
            x = points2[i][0]
            y = points2[i][1]
            try:
                z = points2[i][2]
            except IndexError:
                z = 0.0
            atom_type =  'ALB' if i==0 or i==n-1 else 'ALA'
            atoms += ('{0:} {1:} {2:} {3:} {4:} {5:} {6:}  {7:} {8:} '
                    '{9:} {10:.3f} {11:.3f} {12:.3f}\n'.format('ATOM', n+i+1, 'D', 'CA',\
                                                                '.', 'ALA', 'B', 2, n+i+1, '?',\
                                                                x, y, z))

    connects = ''
    for i in range(0,n-1):
        atom_type0 =  'ALB' if i==0 else 'ALA'
        atom_type1 =  'ALB' if i+1==n-1 else 'ALA'
        connects += f'C{i+1} covale {atom_type0} A {i+1} CA {atom_type1} A {i+2} CA\n'
    if run_repli:
        for i in range(0,n-1):
            atom_type0 =  'ALB' if i==0 else 'ALA'
            atom_type1 =  'ALB' if i+1==n-1 else 'ALA'
            connects += f'C{n+i+1} covale {atom_type0} B {n+i+1} CA {atom_type1} B {n+i+2} CA\n'

    # Save files
    ## .pdb
    cif_file_content = mmcif_atomhead+atoms+mmcif_connecthead+connects

    with open(cif_file_name, 'w') as f:
        f.write(cif_file_content)

def generate_psf(n: int, file_name='replisage.psf', title="No title provided",duplicated=False):
    """
    Saves PSF file. Useful for trajectories in DCD file format.
    :param n: number of points
    :param file_name: PSF file name
    :param title: Human readable string. Required in PSF file.
    :return: List with string records of PSF file.
    """
    assert len(title) < 40, "provided title in psf file is too long."
    # noinspection PyListCreation
    lines = ['PSF CMAP\n']
    lines.append('\n')
    lines.append('      1 !NTITLE\n')
    lines.append('REMARKS {}\n'.format(title))
    lines.append('\n')
    N = n if not duplicated else 2*n
    lines.append('{:>8} !NATOM\n'.format(N))
    for k in range(1, n + 1):
        lines.append('{:>8} BEAD {:<5} ALA  CA   A      0.000000        1.00 0           0\n'.format(k, k))
    if duplicated:
        for k in range(n, 2*n + 1):
            lines.append('{:>8} BEAD {:<5} ALA  CA   B      0.000000        1.00 0           0\n'.format(k, k))
    lines.append('\n')
    lines.append('{:>8} !NBOND: bonds\n'.format(n - 1))
    for i in range(1, n):
        lines.append('{:>8}{:>8}\n'.format(i, i + 1))
    if duplicated:
        for i in range(n+1, 2*n):
            lines.append('{:>8}{:>8}\n'.format(i, i + 1))
    with open(file_name, 'w') as f:
        f.writelines(lines)

############# Computation of heatmaps #############
def get_coordinates_pdb(file:str):
    '''
    It returns the corrdinate matrix V (N,3) of a .pdb file.
    The main problem of this function is that coordiantes are not always in 
    the same column position of a .pdb file. Do changes appropriatelly,
    in case that the data aren't stored correctly. 
    
    Input:
    file (str): the path of the .pdb file.
    
    Output:
    V (numpy array): the matrix of coordinates
    '''
    V = list()
    
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("CONNECT") or line.startswith("END") or line.startswith("TER"):
                break
            if line.startswith("HETATM"): 
                x = float(line[31:38])
                y = float(line[39:46])
                z = float(line[47:54])
                V.append([x, y, z])
    
    return np.array(V)

def get_coordinates_cif(file:str):
    '''
    It returns the corrdinate matrix V (N,3) of a .pdb file.
    The main problem of this function is that coordiantes are not always in 
    the same column position of a .pdb file. Do changes appropriatelly,
    in case that the data aren't stored correctly. 
    
    Input:
    file (str): the path of the .cif file.
    
    Output:
    V (np.array): the matrix of coordinates
    '''
    V = list()
    
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("ATOM"):
                columns = line.split()
                x = eval(columns[10])
                y = eval(columns[11])
                z = eval(columns[12])
                V.append([x, y, z])
    
    return np.array(V)

def get_coordinates_mm(mm_vec):
    '''
    It returns the corrdinate matrix V (N,3) of a .pdb file.
    The main problem of this function is that coordiantes are not always in 
    the same column position of a .pdb file. Do changes appropriatelly,
    in case that the data aren't stored correctly. 
    
    Input:
    file (Openmm Qunatity): an OpenMM vector of the form 
    Quantity(value=[Vec3(x=0.16963918507099152, y=0.9815883636474609, z=-1.4776774644851685), 
    Vec3(x=0.1548253297805786, y=0.9109517931938171, z=-1.4084612131118774), 
    Vec3(x=0.14006929099559784, y=0.8403329849243164, z=-1.3392155170440674), 
    Vec3(x=0.12535107135772705, y=0.7697405219078064, z=-1.269935131072998),
    ...,
    unit=nanometer)
    
    Output:
    V (np.array): the matrix of coordinates
    '''
    V = list()

    for i in range(len(mm_vec)):
        x, y ,z = mm_vec[i][0]._value, mm_vec[i][1]._value, mm_vec[i][2]._value
        V.append([x, y, z])
    
    return np.array(V)

def get_heatmap(mm_vec,save_path=None,th=1,save=False):
    '''
    It returns the corrdinate matrix V (N,3) of a .pdb file.
    The main problem of this function is that coordiantes are not always in 
    the same column position of a .pdb file. Do changes appropriatelly,
    in case that the data aren't stored correctly.
    
    Input:
    file (Openmm Qunatity): an OpenMM vector of the form 
    Quantity(value=[Vec3(x=0.16963918507099152, y=0.9815883636474609, z=-1.4776774644851685),
    Vec3(x=0.1548253297805786, y=0.9109517931938171, z=-1.4084612131118774),
    Vec3(x=0.14006929099559784, y=0.8403329849243164, z=-1.3392155170440674),
    Vec3(x=0.12535107135772705, y=0.7697405219078064, z=-1.269935131072998),
    ...,
    unit=nanometer)
    
    Output:
    H (np.array): a heatmap of the 3D structure.
    '''
    V = get_coordinates_mm(mm_vec)
    mat = distance.cdist(V, V, 'euclidean') # this is the way \--/
    mat = 1/(mat+1)

    if save_path!=None:
        figure(figsize=(25, 20))
        plt.imshow(mat,cmap="Reds")
        if save: plt.savefig(save_path,format='svg',dpi=500)
        plt.close()
        if save: np.save(save_path.replace("svg", "npy"),mat)
    return mat

def heats_to_prob(heats,path,burnin,q=0.15):
    q_dist = np.quantile(np.array(heats),1-q)
    prob_mat = np.zeros(heats[0].shape)

    norm = np.zeros(len(heats[0]))
    for heat in heats:
        for i in range(len(heats[0])):
            norm[i]+=(np.average(np.diagonal(heat,offset=i))+np.average(np.diagonal(heat,offset=-i)))/2
    norm = norm/len(heats)

    for i in range(burnin,len(heats)):
        prob_mat[heats[i]>q_dist] += 1
    
    prob_mat = prob_mat/len(heats)
    for i in range(len(prob_mat)):
        for j in range(0,len(prob_mat)-i):
            prob_mat[i,i+j]=prob_mat[i,i+j]/norm[j]
            prob_mat[i+j,i]=prob_mat[i+j,i]/norm[j]
    
    figure(figsize=(10, 10))
    plt.imshow(prob_mat,cmap="Reds")
    plt.colorbar()
    plt.title(f'Normalized Probability distribution that distance < {q} quantile', fontsize=13)
    plt.savefig(path,format='png',dpi=500)
    plt.show(block=False)

def binned_distance_matrix(idx,folder_name,input=None,th=23):
    '''
    This function calculates the mean distance through models, between two specific beads.
    We do that for all the possible beads and we take a matrix/heatmap.
    This one may take some hours for many beads or models.
    This works for .pdb files.
    '''
    
    V = get_coordinates_pdb(folder_name+f'/pdbs/SM{idx}.pdb')
    mat = distance.cdist(V, V, 'euclidean') # this is the way \--/ 

    figure(figsize=(25, 20))
    plt.imshow(mat,cmap=LinearSegmentedColormap.from_list("bright_red",[(1,0,0),(1,1,1)]), vmin=0, vmax=th)
    plt.savefig(folder_name+f'/heatmaps/SM_bindist_heatmap_idx{idx}.png',format='png',dpi=500)
    plt.close()

    np.save(folder_name+f'/heatmaps/binned_dist_matrix_idx{idx}.npy',mat)
    
    return mat

def average_binned_distance_matrix(folder_name,N_steps,step,burnin,th1=0,th2=23):
    '''
    This function calculates the mean distance through models, between two specific beads.
    We do that for all the possible beads and we take a matrix/heatmap.
    This one may take some hours for many beads or models.
    smoothing (str): You can choose between 'Nearest Neighbour', 'bilinear', 'hanning', 'bicubic'.
    '''
    sum_mat = 0
    for i in tqdm(range(0,N_steps,step)):
        V = get_coordinates_pdb(folder_name+f'/pdbs/SM{i}.pdb')
        if i >= burnin*step:
            sum_mat += distance.cdist(V, V, 'euclidean') # this is the way \--/ 
    new_N = N_steps//step
    avg_mat = sum_mat/new_N
    
    figure(figsize=(25, 20))
    plt.imshow(avg_mat,cmap=LinearSegmentedColormap.from_list("bright_red",[(1,0,0),(1,1,1)]), vmin=th1, vmax=th2)
    plt.savefig(folder_name+f'/plots/SM_avg_bindist_heatmap.png',format='png',dpi=500)
    plt.show(block=False)
    np.save(folder_name+'/plots/average_binned_dist_matrix.npy',avg_mat)

    return avg_mat

########## Statistics ###########
def get_stats(ms,ns,N_beads):
    '''
    This is a function that computes maximum compaction score in every step of the simulation.
    '''
    # Computing Folding Metrics
    N_coh = len(ms)
    chromatin = np.zeros(N_beads)
    chromatin2 = np.zeros(N_beads)
    for nn in range(N_coh):
        m,n = int(ms[nn]),int(ns[nn])
        if m<=n:
            chromatin[m:n] = 1
            chromatin2[m:n]+=1
        else:
            chromatin[0:n] = 1
            chromatin[m:] = 1
            chromatin2[0:n]+=1
            chromatin2[m:]+=1
    f = np.mean(chromatin)
    F = np.mean(chromatin2)
    f_std = np.std(chromatin)
    FC = 1/(1-f+0.001)
    
    return f, f_std, F, FC

def get_avg_heatmap(path, N1, N2, method='mean_heatmap', epsilon=1e-6):
    """
    Computes inverse distance heatmaps for an ensemble of 3D structures.

    Parameters:
    -----------
    path : str
        Path to the directory containing CIF structure files in /ensemble/.
    N1 : int
        Starting index (inclusive) of ensemble structures to include.
    N2 : int
        Ending index (exclusive) of ensemble structures to include.
    method : str
        Either 'mean_structure' or 'mean_heatmap':
            - 'mean_structure': computes inverse distance map of the average 3D structure.
                * Fast and low-memory.
                * Less accurate for experimental comparison due to nonlinearity of distance.
            - 'mean_heatmap': accumulates and averages inverse distance maps over structures.
                * Best match for population-averaged experimental data (e.g., Hi-C, FISH).
                * Now optimized to be memory-efficient.
    epsilon : float
        Small constant to avoid division by zero.

    Returns:
    --------
    mean_heat : np.ndarray
        Averaged inverse distance heatmap.
    std_heat : np.ndarray
        Inverse distance heatmap of standard deviation structure.
    """
    os.makedirs(path + '/metadata/structural_metrics', exist_ok=True)
    os.makedirs(path + '/plots/structural_metrics', exist_ok=True)

    # Try loading one structure to get N_beads
    first_file = glob.glob(path + f'/ensemble/ensemble_{N1}_*.cif')
    if not first_file:
        print(f"No starting structure found at ensemble_{N1}_*.cif")
        return None, None
    V_init = get_coordinates_cif(first_file[0])
    N_beads = len(V_init) // 2

    structure_sum = np.zeros((N_beads, 3), dtype=np.float64)
    structure_sq_sum = np.zeros((N_beads, 3), dtype=np.float64)
    heatmap_sum = np.zeros((N_beads, N_beads), dtype=np.float64)
    count = 0

    for i in tqdm(range(N1, N2)):
        file_list = glob.glob(path + f'/ensemble/ensemble_{i}_*.cif')
        if not file_list:
            print(f"Warning: No file for index {i}")
            continue
        try:
            V = get_coordinates_cif(file_list[0])[:N_beads]
            structure_sum += V
            structure_sq_sum += V**2
            count += 1

            if method == 'mean_heatmap':
                D = distance.cdist(V, V, 'euclidean')
                heatmap_sum += 1.0 / (D + epsilon)

        except Exception as e:
            print(f"Error reading {file_list[0]}: {e}")
            continue

    if count == 0:
        print("No valid structures loaded.")
        return None, None

    # Finalize structures
    mean_V = structure_sum / count
    std_V = np.sqrt(structure_sq_sum / count - mean_V**2)

    np.save(path + f'/metadata/structural_metrics/mean_structure_{N1}_{N2}.npy', mean_V)
    np.save(path + f'/metadata/structural_metrics/std_structure_{N1}_{N2}.npy', std_V)

    # Compute inverse heatmaps
    if method == 'mean_structure':
        D_mean = distance.cdist(mean_V, mean_V, 'euclidean')
        mean_heat = 1.0 / (D_mean + epsilon)
    else:  # 'mean_heatmap'
        mean_heat = heatmap_sum / count

    D_std = distance.cdist(std_V, std_V, 'euclidean')
    std_heat = 1.0 / (D_std + epsilon)

    # Save heatmaps
    np.save(path + f'/metadata/structural_metrics/mean_inv_heatmap_{N1}_{N2}.npy', mean_heat)
    np.save(path + f'/metadata/structural_metrics/std_inv_heatmap_{N1}_{N2}.npy', std_heat)

    # Plot
    try:
        figure(figsize=(20, 20))
        plt.imshow(mean_heat, cmap='viridis', aspect='auto')
        plt.title('Mean Inverse Distance Heatmap')
        plt.colorbar()
        plt.savefig(path + f'/plots/structural_metrics/mean_inv_heatmap_{N1}_{N2}.png', dpi=200)
        plt.savefig(path + f'/plots/structural_metrics/mean_inv_heatmap_{N1}_{N2}.svg', dpi=200)
        plt.close()

        figure(figsize=(20, 20))
        plt.imshow(std_heat, cmap='inferno', aspect='auto')
        plt.title('STD Inverse Distance Heatmap')
        plt.colorbar()
        plt.savefig(path + f'/plots/structural_metrics/std_inv_heatmap_{N1}_{N2}.png', dpi=200)
        plt.savefig(path + f'/plots/structural_metrics/std_inv_heatmap_{N1}_{N2}.svg', dpi=200)
        plt.close()
    except Exception as e:
        print(f"Plotting failed: {e}")

    return mean_heat, std_heat