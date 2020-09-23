#!/usr/bin/env bash

# make sure anaconda environment is loaded
python -m rigid_DL.cp_qe_test 80_ele_meshes/211_quad.vtk 2 1 1 -o 211_cp_qe &
python -m rigid_DL.cp_qe_test 80_ele_meshes/221_quad.vtk 2 2 1 -o 221_cp_qe &
python -m rigid_DL.cp_qe_test 80_ele_meshes/411_quad.vtk 4 1 1 -o 411_cp_qe &
python -m rigid_DL.cp_qe_test 80_ele_meshes/441_quad.vtk 4 4 1 -o 441_cp_qe &
python -m rigid_DL.cp_qe_test 80_ele_meshes/811_quad.vtk 8 1 1 -o 811_cp_qe &
python -m rigid_DL.cp_qe_test 80_ele_meshes/881_quad.vtk 8 8 1 -o 881_cp_qe &

python -m rigid_DL.lp_qe_test 80_ele_meshes/211_quad.vtk 2 1 1 -o 211_lp_qe &
python -m rigid_DL.lp_qe_test 80_ele_meshes/221_quad.vtk 2 2 1 -o 221_lp_qe &
python -m rigid_DL.lp_qe_test 80_ele_meshes/411_quad.vtk 4 1 1 -o 411_lp_qe &
python -m rigid_DL.lp_qe_test 80_ele_meshes/441_quad.vtk 4 4 1 -o 441_lp_qe &
python -m rigid_DL.lp_qe_test 80_ele_meshes/811_quad.vtk 8 1 1 -o 811_lp_qe &
python -m rigid_DL.lp_qe_test 80_ele_meshes/881_quad.vtk 8 8 1 -o 881_lp_qe &
