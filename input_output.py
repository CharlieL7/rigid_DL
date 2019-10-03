"""
Module to read and write meshes to Tecplot data file type
extended from tec_dat module
"""

import csv
import sys
import numpy as np


def read_dat(filename):
    """
     Reads in vesicle shape dat files for tecplot into python lists.

     Parameters:
        filename : the name of the file as a string
     Returns:
        (all_data, f2v, params)
        all_data : data for each vertex, first 3 will always be position
        f2v : connectivity data
        params : dict of all other values taken from file
    """
    with open(filename, 'r') as dat_file:
    # reading in the header block
        is_header = True
        header_block = []
        while is_header:
            tmp_line = dat_file.readline()
            if tmp_line.find('#') == 0:
                header_block.append(tmp_line)
                eq_pos = tmp_line.find('=')
                if tmp_line.find("De") != -1:
                    De = float(tmp_line[eq_pos+1:])
                elif tmp_line.find("time") != -1:
                    time = float(tmp_line[eq_pos+1:])
                elif tmp_line.find("viscRat") != -1:
                    visc_rat = float(tmp_line[eq_pos+1:])
                elif tmp_line.find("volRat") != -1:
                    vol_rat = float(tmp_line[eq_pos+1:])
                elif tmp_line.find("deformRate") != -1:
                    deformRate = float(tmp_line[eq_pos+1:])
                elif tmp_line.find("tnum") != -1:
                    _tnum = float(tmp_line[eq_pos+1:])
            else:
                is_header = False
                # get the next two Tecplot lines and then go back one line
                header_block.append(ggtmp_line)
                last_pos = dat_file.tell()
                tmp_line = dat_file.readline()
                header_block.append(tmp_line)
                dat_file.seek(last_pos)

        try:
            reader = csv.reader(dat_file, delimiter=' ')
            type_line = next(reader)
            nvert = int(type_line[1][2:])
            nface = int(type_line[2][2:])
            all_data = [] # position + all other data
            f2v = [] # connectivity

            count = 0
            while count < nvert:
                lst = next(reader)
                all_data.append(lst)
                count += 1
            all_data = np.array(all_data, dtype=float)

            count = 0
            while count < nface:
                lst = next(reader)[0:3] # should just be 3 values
                f2v.append([int(i) for i in lst])
                count += 1
            f2v = np.array(f2v, dtype=int)

        except csv.Error as e:
            sys.exit('file %s, line %d: %s' % (filename, reader.line_num, e))

    try:
        params = {"visc_rat": visc_rat, "vol_rat": vol_rat, "De": De, "deformRate": deformRate,
                  "time": time, "nvert": nvert, "nface": nface, "header_block": header_block}
        return (all_data, f2v, params)
    except NameError as e:
        print("One of the required variables was not instantiated: {}".format(e))


def read_short_dat(filename):
    """
     Reads in vesicle shape dat files for tecplot into python lists.
     This version only reads the positions and connectivity

     Parameters:
        filename: the name of the file as a string
     Returns:
        (all_data, f2v, params)
        all_data: data for each vertex, first 3 will always be position
        f2v: connectivity data
        params: only header_block
    """
    with open(filename, 'r') as dat_file:
    # reading in the header block
        is_header = True
        header_block = []
        while is_header:
            tmp_line = dat_file.readline()
            if tmp_line.find('#') == 0:
                header_block.append(tmp_line)
            else:
                is_header = False
                # get the next two Tecplot lines and then go back one line
                header_block.append(tmp_line)
                last_pos = dat_file.tell()
                tmp_line = dat_file.readline()
                header_block.append(tmp_line)
                dat_file.seek(last_pos)

        try:
            reader = csv.reader(dat_file, delimiter=' ')
            type_line = next(reader)
            nvert = int(type_line[1][2:])
            nface = int(type_line[2][2:])
            all_data = [] # position + all other data
            f2v = [] # connectivity

            count = 0
            while count < nvert:
                lst = next(reader)
                all_data.append(lst)
                count += 1
            all_data = np.array(all_data, dtype=float)

            count = 0
            while count < nface:
                lst = next(reader)[0:3] # should just be 3 values
                f2v.append([int(i) for i in lst])
                count += 1
            f2v = np.array(f2v, dtype=int)

        except csv.Error as e:
            sys.exit('file %s, line %d: %s' % (filename, reader.line_num, e))

    try:
        params = {"header_block": header_block}
        return (all_data, f2v, params)
    except NameError as e:
        print("One of the required variables was not instantiated: {}".format(e))


def write_dat(all_data, f2v, params, out_name):
    """
     Writes vesicle shape data into a tecplot readable .dat format.

     Parameters:
        all_data : postition + all other data
        f2v : connectivity data as 2D numpy array
        params : dict of all other simulation parameters
        out_name : string of the filename you want for the new file
     Returns:
         None
    """
    with open(out_name, 'w') as out:
        out.writelines(params["header_block"])
        writer = csv.writer(out, delimiter=' ', lineterminator="\n")
        writer.writerows(all_data)
        writer.writerows(f2v)


def calc_moment_inertia_tensor(x):
    """
    Calculates the moment of inertia tensor for a 2D numpy array of coordinates.
    Uses equal weighting.

    Parameters:
        x : ndarray of vertex positions
    Returns:
        tuple of (moment of inertia tensor, the eigenvalues, and eigenvectors)
    """
    (nvert, __) = x.shape
    inertia_tensor = np.zeros((3, 3))
    delta_ij = np.identity(3)
    for i in range(0, nvert):
        r_vec = x[i, :] # positions
        inertia_tensor += np.dot(r_vec, r_vec) * delta_ij - np.outer(r_vec, r_vec)
    [eigvals, eigvecs] = np.linalg.eig(inertia_tensor)
    idx = eigvals.argsort()
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # first vector positive x
    if eigvecs[0, 0] < 0:
        eigvecs[:, 0] = -eigvecs[:, 0]

    # making sure right-handed
    temp = np.cross(eigvecs[:, 0], eigvecs[:, 1])
    if np.dot(temp.T, eigvecs[:, 2]) < 0:
        eigvecs[:, 2] = -eigvecs[:, 2]

    return (inertia_tensor, eigvals, eigvecs)
