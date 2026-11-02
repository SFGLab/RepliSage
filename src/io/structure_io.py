"""Read/write 3D structures (mmCIF, PSF, CMM, PDB)."""
import numpy as np

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
_atom_site.B_iso_or_equiv
_atom_site.auth_asym_id
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

def write_cif(V, n_chains=1, path='structure.cif', bfactor=None, element='C'):
    """ChimeraX-ready mmCIF: one chain per DNA copy (A, B), CA-only beads, covalent bonds, B-factor = position along the chain."""
    V = np.asarray(V, float); n = len(V) // n_chains
    if bfactor is None: bfactor = np.tile(np.linspace(0, 100, n), n_chains)
    rows, bonds = [], []
    for c in range(n_chains):
        ch = "AB"[c]
        for i in range(n):
            k = c * n + i; x, y, z = V[k]
            rows.append(f"ATOM {k + 1} {element} CA . ALA {ch} {c + 1} {i + 1} ? {x:.3f} {y:.3f} {z:.3f} {bfactor[k]:.2f} {ch}")
        bonds += [f"C{c * n + i + 1} covale ALA {ch} {i + 1} CA ALA {ch} {i + 2} CA" for i in range(n - 1)]
    with open(path, 'w') as f:
        f.write(mmcif_atomhead + "\n".join(rows) + "\n" + mmcif_connecthead + "\n".join(bonds) + "\n")

def write_mmcif(points1, points2=None, cif_file_name='LE_init_struct.cif'):
    """Initial structure (one or two chains) as mmCIF."""
    V = np.asarray(points1, float) if points2 is None else np.vstack([points1, points2])
    write_cif(V, 1 if points2 is None else 2, cif_file_name, element='D')   # 'D': matches the force-field template

def write_chimerax_script(ensemble_dir, files, n_chains=1):
    """view.cxc: opens the ensemble as a trajectory, tube style, sisters orange / gold or rainbow."""
    import os
    cmd = ["# ChimeraX: File > Open this script (or: open view.cxc)", f"open {files[0]}",
           "hide cartoons; show atoms; style ball; size ballScale 0.55; size pseudobondRadius 0.4",
           "color /A orange target abp; color /B gold target abp" if n_chains == 2 else "color bfactor palette rainbow target abp",
           "lighting soft; graphics silhouettes true; set bgColor white; view"]
    with open(os.path.join(ensemble_dir, "view.cxc"), "w") as fh: fh.write("\n".join(cmd) + "\n")

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
                x = float(columns[10])
                y = float(columns[11])
                z = float(columns[12])
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
