import numpy as np
import torch

from irreps_utils import _cartesian_matrix_to_irreps, _CARTESIAN, _irreps_to_cartesian_matrix, irreps_to_symmetric_tensor, _SYMMETRIC_CARTESIAN 
from transformations import rotate_3d, get_all_octahedral_angles

x = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
x = x.unsqueeze(0).unsqueeze(0).unsqueeze(0)
x_rot = rotate_3d(x, get_all_octahedral_angles()[17])

x_irreps = _cartesian_matrix_to_irreps(x[0][0], _CARTESIAN)
x_rot_irreps = _cartesian_matrix_to_irreps(x_rot[0][0], _CARTESIAN)

x_inverse = _irreps_to_cartesian_matrix(x_irreps, _CARTESIAN)
x_rot_inverse = _irreps_to_cartesian_matrix(x_rot_irreps, _CARTESIAN)

print(x)
print(x_inverse)
print(torch.allclose(x_inverse[0], x[0][0][0], rtol=1e-5, atol=1e-8))


print(x_rot)
print(x_rot_inverse)
print(torch.allclose(x_rot_inverse[0], x_rot[0][0][0], rtol=1e-5, atol=1e-8))

x = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
x = x.unsqueeze(0).unsqueeze(0).unsqueeze(0)
x_rot = rotate_3d(x, get_all_octahedral_angles()[2])

x_irreps = _cartesian_matrix_to_irreps(x[0][0], _CARTESIAN)
x_rot_irreps = _cartesian_matrix_to_irreps(x_rot[0][0], _CARTESIAN)

x_inverse = _irreps_to_cartesian_matrix(x_irreps, _CARTESIAN)
x_rot_inverse = _irreps_to_cartesian_matrix(x_rot_irreps, _CARTESIAN)

print(x)
print(x_inverse)
print(torch.allclose(x_inverse[0], x[0][0][0], rtol=1e-5, atol=1e-8))

print(x_rot)
print(x_rot_inverse)
print(torch.allclose(x_rot_inverse[0], x_rot[0][0][0], rtol=1e-5, atol=1e-8))


x = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
x = x.unsqueeze(0).unsqueeze(0).unsqueeze(0)
x_rot = rotate_3d(x, get_all_octahedral_angles()[2])

x_irreps = _cartesian_matrix_to_irreps(x[0][0], _CARTESIAN)
x_rot_irreps = _cartesian_matrix_to_irreps(x_rot[0][0], _CARTESIAN)

x_inverse = _irreps_to_cartesian_matrix(x_irreps, _CARTESIAN)
x_rot_inverse = _irreps_to_cartesian_matrix(x_rot_irreps, _CARTESIAN)

print(x)
print(x_inverse)
print(torch.allclose(x_inverse[0], x[0][0][0], rtol=1e-5, atol=1e-8))

print(x_rot)
print(x_rot_inverse)
print(torch.allclose(x_rot_inverse[0], x_rot[0][0][0], rtol=1e-5, atol=1e-8))

x = torch.tensor([[1., 2., 3.], [2., 5., 6.], [3., 6., 9.]])
x = x.unsqueeze(0).unsqueeze(0).unsqueeze(0)
x_rot = rotate_3d(x, get_all_octahedral_angles()[2])

x_irreps = _cartesian_matrix_to_irreps(x[0][0], _SYMMETRIC_CARTESIAN)
x_rot_irreps = _cartesian_matrix_to_irreps(x_rot[0][0], _CARTESIAN)

x_inverse = _irreps_to_cartesian_matrix(x_irreps, _SYMMETRIC_CARTESIAN)
x_rot_inverse = _irreps_to_cartesian_matrix(x_rot_irreps, _CARTESIAN)

print(x)
print(x_inverse)
print(torch.allclose(x_inverse[0], x[0][0][0], rtol=1e-5, atol=1e-8))

print(x_rot)
print(x_rot_inverse)
print(torch.allclose(x_rot_inverse[0], x_rot[0][0][0], rtol=1e-5, atol=1e-8))


x = torch.tensor([[1., 2., 3.], [2., 5., 6.], [3., 6., 9.]])
x = x.unsqueeze(0).unsqueeze(0).unsqueeze(0)
x_rot = rotate_3d(x, get_all_octahedral_angles()[2])

x_irreps = _cartesian_matrix_to_irreps(x[0][0], _SYMMETRIC_CARTESIAN)



print(x)
