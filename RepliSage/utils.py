#########################################################################
########### CREATOR: SEBASTIAN KORSAK, WARSAW 2022 ######################
#########################################################################

from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import pandas as pd
from scipy.spatial import distance
from importlib.resources import files
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats.stats import pearsonr, spearmanr, kendalltau
from tqdm import tqdm

def make_folder(folder_name):
    try:
        os.mkdir(folder_name)
        os.mkdir(folder_name+'/plots')
        os.mkdir(folder_name+'/other')
        os.mkdir(folder_name+'/ensemble')
    except OSError as error:
        print(f'Directory with name "{folder_name}" already exists! Directory will be overwritten.')
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

def get_avg_heatmap(path,N1,N2):
    file_pattern = path + f'/ensemble/ensemble_1_*.cif'
    file_list = glob.glob(file_pattern)
    V = get_coordinates_cif(file_list[0])
    N_beads = len(V)//2
    avg_heat = np.zeros((N_beads,N_beads))
    for i in tqdm(range(N1,N2)):
        file_pattern = path + f'/ensemble/ensemble_{i}_*.cif'
        file_list = glob.glob(file_pattern)
        V = get_coordinates_cif(file_list[0])[:N_beads]
        heat =  distance.cdist(V, V, 'euclidean') # this is the way \--/
        avg_heat += 1/heat

    avg_heat = avg_heat/(N2-N1)
    np.save(path+f'/other/heatmap_{N1}_{N2}.npy', avg_heat)

    figure(figsize=(20, 20))
    plt.imshow(avg_heat,cmap='Reds',vmax=0.2, aspect='auto')
    plt.savefig(path+f'/plots/heatmap_{N1}_{N2}.png',format='png',dpi=200)
    plt.savefig(path+f'/plots/heatmap_{N1}_{N2}.svg',format='svg',dpi=200)
    plt.close()