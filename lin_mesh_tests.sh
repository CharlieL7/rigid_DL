#!/usr/bin/env bash

# make sure anaconda environment is loaded
python -m rigid_DL.cp_le_test 80_ele_meshes/211_lin.vtk 2 1 1 -o 211_cp_le &
python -m rigid_DL.cp_le_test 80_ele_meshes/221_lin.vtk 2 2 1 -o 221_cp_le &
python -m rigid_DL.cp_le_test 80_ele_meshes/411_lin.vtk 4 1 1 -o 411_cp_le &
python -m rigid_DL.cp_le_test 80_ele_meshes/441_lin.vtk 4 4 1 -o 441_cp_le &
python -m rigid_DL.cp_le_test 80_ele_meshes/811_lin.vtk 8 1 1 -o 811_cp_le &
python -m rigid_DL.cp_le_test 80_ele_meshes/881_lin.vtk 8 8 1 -o 881_cp_le &

python -m rigid_DL.lp_le_test 80_ele_meshes/211_lin.vtk 2 1 1 -o 211_lp_le &
python -m rigid_DL.lp_le_test 80_ele_meshes/221_lin.vtk 2 2 1 -o 221_lp_le &
python -m rigid_DL.lp_le_test 80_ele_meshes/411_lin.vtk 4 1 1 -o 411_lp_le &
python -m rigid_DL.lp_le_test 80_ele_meshes/441_lin.vtk 4 4 1 -o 441_lp_le &
python -m rigid_DL.lp_le_test 80_ele_meshes/811_lin.vtk 8 1 1 -o 811_lp_le &
python -m rigid_DL.lp_le_test 80_ele_meshes/881_lin.vtk 8 8 1 -o 881_lp_le &
