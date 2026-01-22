import torch
import sys
import pytest

sys.path.append("./src/")
from transformations import euler_angles_to_matrix, get_all_octahedral_angles, rotate_octahedral_exact


# ============ Helper functions ============

def tensor6_to_matrix(t):
    """Convert 6-channel tensor [xx, yy, zz, xy, xz, yz] to 3x3 symmetric matrix."""
    T = torch.zeros(3, 3)
    T[0, 0], T[1, 1], T[2, 2] = t[0], t[1], t[2]
    T[0, 1] = T[1, 0] = t[3]
    T[0, 2] = T[2, 0] = t[4]
    T[1, 2] = T[2, 1] = t[5]
    return T


def matrix_to_tensor6(T):
    """Convert 3x3 symmetric matrix to 6-channel tensor."""
    return torch.tensor([T[0, 0], T[1, 1], T[2, 2], T[0, 1], T[0, 2], T[1, 2]])


def tensor9_to_matrix(t):
    """Convert 9-channel tensor to 3x3 matrix (row-major order)."""
    return t.reshape(3, 3)


def matrix_to_tensor9(T):
    """Convert 3x3 matrix to 9-channel tensor."""
    return T.flatten()


def rotate_tensor(T, R):
    """Rotate a 3x3 tensor: T' = R @ T @ R.T"""
    return R @ T @ R.T


def find_inverse_rotation(matrices, i):
    """Find index of inverse rotation for rotation i."""
    R_inv = matrices[i].T
    for k in range(len(matrices)):
        if torch.allclose(matrices[k], R_inv, atol=1e-5):
            return k
    return None


def find_composed_rotation(matrices, i, j):
    """Find index of composed rotation R_j @ R_i."""
    R_composed = matrices[j] @ matrices[i]
    R_composed_int = torch.round(R_composed).int()
    for k in range(len(matrices)):
        if torch.allclose(torch.round(matrices[k]).int().float(), R_composed_int.float()):
            return k
    return None


# ============ Vector field tests (C=3) ============

class TestVectorField:

    def test_uniform_x_vector_rotation(self):
        """Test that uniform +X vectors are rotated correctly through all 24 octahedral rotations."""
        field = torch.zeros(1, 3, 2, 2, 2)
        field[:, 0] = 1.0

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            angle = angles[i:i+1]
            R = matrices[i]
            expected = R[:, 0]

            rotated = rotate_octahedral_exact(field, angle, rotate_channels=True, mode="pairwise")
            actual = rotated[0, :, 0, 0, 0]

            assert torch.allclose(actual, expected, atol=1e-5), f"Rotation {i}: expected={expected.tolist()}, actual={actual.tolist()}"

    def test_uniform_y_vector_rotation(self):
        """Test that uniform +Y vectors are rotated correctly."""
        field = torch.zeros(1, 3, 2, 2, 2)
        field[:, 1] = 1.0

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            angle = angles[i:i+1]
            expected = matrices[i][:, 1]

            rotated = rotate_octahedral_exact(field, angle, rotate_channels=True, mode="pairwise")
            actual = rotated[0, :, 0, 0, 0]

            assert torch.allclose(actual, expected, atol=1e-5), f"Rotation {i}"

    def test_uniform_z_vector_rotation(self):
        """Test that uniform +Z vectors are rotated correctly."""
        field = torch.zeros(1, 3, 2, 2, 2)
        field[:, 2] = 1.0

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            angle = angles[i:i+1]
            expected = matrices[i][:, 2]

            rotated = rotate_octahedral_exact(field, angle, rotate_channels=True, mode="pairwise")
            actual = rotated[0, :, 0, 0, 0]

            assert torch.allclose(actual, expected, atol=1e-5), f"Rotation {i}"

    def test_negative_x_vector_rotation(self):
        """Test that uniform -X vectors are rotated correctly."""
        field = torch.zeros(1, 3, 2, 2, 2)
        field[:, 0] = -1.0

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            angle = angles[i:i+1]
            expected = -matrices[i][:, 0]

            rotated = rotate_octahedral_exact(field, angle, rotate_channels=True, mode="pairwise")
            actual = rotated[0, :, 0, 0, 0]

            assert torch.allclose(actual, expected, atol=1e-5), f"Rotation {i}"

    def test_diagonal_vector_rotation(self):
        """Test that uniform [1,1,1] vectors are rotated correctly."""
        field = torch.zeros(1, 3, 2, 2, 2)
        field[:, 0] = 1.0
        field[:, 1] = 1.0
        field[:, 2] = 1.0

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            angle = angles[i:i+1]
            expected = matrices[i] @ torch.tensor([1.0, 1.0, 1.0])

            rotated = rotate_octahedral_exact(field, angle, rotate_channels=True, mode="pairwise")
            actual = rotated[0, :, 0, 0, 0]

            assert torch.allclose(actual, expected, atol=1e-5), f"Rotation {i}"

    def test_spatial_permutation(self):
        """Test that spatial positions are permuted correctly through all 24 octahedral rotations."""
        field = torch.zeros(1, 3, 2, 2, 2)
        field[0, 0, 0, 0, 0] = 1.0

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            angle = angles[i:i+1]
            R = torch.round(matrices[i]).int()

            rotated = rotate_octahedral_exact(field, angle, rotate_channels=True, mode="pairwise")

            actual_pos = None
            for d in range(2):
                for h in range(2):
                    for w in range(2):
                        if rotated[0, :, d, h, w].abs().max() > 0.5:
                            actual_pos = (d, h, w)

            coord = torch.tensor([-0.5, -0.5, -0.5], dtype=torch.float32)
            new_coord = (R.float() @ coord + 0.5).clamp(0, 1)
            expected_pos = tuple(new_coord.int().tolist())

            assert actual_pos == expected_pos, f"Rotation {i}: actual={actual_pos}, expected={expected_pos}"

    def test_multiple_positions(self):
        """Test that vectors at two corners are both transformed correctly."""
        field = torch.zeros(1, 3, 2, 2, 2)
        field[0, 0, 0, 0, 0] = 1.0
        field[0, 1, 1, 1, 1] = 1.0

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            angle = angles[i:i+1]

            rotated = rotate_octahedral_exact(field, angle, rotate_channels=True, mode="pairwise")

            expected_vec1 = matrices[i][:, 0]
            expected_vec2 = matrices[i][:, 1]

            found_vec1, found_vec2 = False, False
            for d in range(2):
                for h in range(2):
                    for w in range(2):
                        vec = rotated[0, :, d, h, w]
                        if torch.allclose(vec, expected_vec1, atol=1e-5):
                            found_vec1 = True
                        if torch.allclose(vec, expected_vec2, atol=1e-5):
                            found_vec2 = True

            assert found_vec1, f"Rotation {i}: didn't find rotated +X vector"
            assert found_vec2, f"Rotation {i}: didn't find rotated +Y vector"

    def test_identity_rotation(self):
        """Test that identity rotation leaves field unchanged."""
        field = torch.zeros(1, 3, 2, 2, 2)
        field[0, 0, 0, 0, 0] = 1.0
        field[0, 1, 0, 1, 0] = 2.0
        field[0, 2, 1, 0, 1] = 3.0

        angles = get_all_octahedral_angles()
        identity_angle = angles[0:1]

        rotated = rotate_octahedral_exact(field, identity_angle, rotate_channels=True, mode="pairwise")

        assert torch.allclose(rotated, field, atol=1e-5)

    def test_rotation_composition(self):
        """Test that applying two rotations equals applying their composition."""
        field = torch.zeros(1, 3, 2, 2, 2)
        field[0, 0, 0, 0, 0] = 1.0

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        i, j = 5, 11
        angle_i, angle_j = angles[i:i+1], angles[j:j+1]

        rotated_once = rotate_octahedral_exact(field, angle_i, rotate_channels=True, mode="pairwise")
        rotated_twice = rotate_octahedral_exact(rotated_once, angle_j, rotate_channels=True, mode="pairwise")

        composed_idx = find_composed_rotation(matrices, i, j)
        assert composed_idx is not None, "Composed rotation not found in group"

        rotated_composed = rotate_octahedral_exact(field, angles[composed_idx:composed_idx+1], rotate_channels=True, mode="pairwise")

        assert torch.allclose(rotated_twice, rotated_composed, atol=1e-5)

    def test_inverse_rotations(self):
        """Test that applying a rotation and its inverse returns the original field."""
        field = torch.zeros(1, 3, 2, 2, 2)
        field[0, 0, 0, 0, 0] = 1.0
        field[0, 1, 1, 0, 1] = 2.0

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            angle_i = angles[i:i+1]

            inv_idx = find_inverse_rotation(matrices, i)
            assert inv_idx is not None, f"Inverse of rotation {i} not found"

            rotated = rotate_octahedral_exact(field, angle_i, rotate_channels=True, mode="pairwise")
            restored = rotate_octahedral_exact(rotated, angles[inv_idx:inv_idx+1], rotate_channels=True, mode="pairwise")

            assert torch.allclose(restored, field, atol=1e-5), f"Rotation {i}"


# ============ Vector field stress tests (C=3) ============

class TestVectorFieldStress:

    def test_random_field_inverse(self):
        """Test that R^{-1}(R(field)) = field for random fields."""
        torch.manual_seed(42)
        field = torch.randn(1, 3, 4, 4, 4)

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            inv_idx = find_inverse_rotation(matrices, i)
            assert inv_idx is not None

            rotated = rotate_octahedral_exact(field, angles[i:i+1], rotate_channels=True, mode="pairwise")
            restored = rotate_octahedral_exact(rotated, angles[inv_idx:inv_idx+1], rotate_channels=True, mode="pairwise")

            assert torch.allclose(restored, field, atol=1e-5), f"Rotation {i}"

    def test_random_field_composition(self):
        """Test that R_j(R_i(field)) = R_{ji}(field) for random fields."""
        torch.manual_seed(43)
        field = torch.randn(1, 3, 4, 4, 4)

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(0, len(angles), 4):
            for j in range(0, len(angles), 4):
                composed_idx = find_composed_rotation(matrices, i, j)
                assert composed_idx is not None

                rotated_twice = rotate_octahedral_exact(
                    rotate_octahedral_exact(field, angles[i:i+1], rotate_channels=True, mode="pairwise"),
                    angles[j:j+1], rotate_channels=True, mode="pairwise"
                )
                rotated_composed = rotate_octahedral_exact(field, angles[composed_idx:composed_idx+1], rotate_channels=True, mode="pairwise")

                assert torch.allclose(rotated_twice, rotated_composed, atol=1e-5), f"Rotations {i}, {j}"

    def test_random_field_norm_preservation(self):
        """Test that rotation preserves the L2 norm of the field."""
        torch.manual_seed(44)
        field = torch.randn(1, 3, 4, 4, 4)
        original_norm = torch.norm(field)

        angles = get_all_octahedral_angles()

        for i in range(len(angles)):
            rotated = rotate_octahedral_exact(field, angles[i:i+1], rotate_channels=True, mode="pairwise")
            rotated_norm = torch.norm(rotated)

            assert torch.allclose(original_norm, rotated_norm, atol=1e-5), f"Rotation {i}"

    def test_random_field_vector_magnitude_preservation(self):
        """Test that rotation preserves the sum of vector magnitudes."""
        torch.manual_seed(45)
        field = torch.randn(1, 3, 4, 4, 4)
        original_magnitude_sum = torch.norm(field, dim=1).sum()

        angles = get_all_octahedral_angles()

        for i in range(len(angles)):
            rotated = rotate_octahedral_exact(field, angles[i:i+1], rotate_channels=True, mode="pairwise")
            rotated_magnitude_sum = torch.norm(rotated, dim=1).sum()

            assert torch.allclose(original_magnitude_sum, rotated_magnitude_sum, atol=1e-5), f"Rotation {i}"

    def test_random_field_dot_product_preservation(self):
        """Test that rotation preserves sum of dot products between two fields."""
        torch.manual_seed(46)
        field1 = torch.randn(1, 3, 4, 4, 4)
        field2 = torch.randn(1, 3, 4, 4, 4)
        original_dot_sum = (field1 * field2).sum()

        angles = get_all_octahedral_angles()

        for i in range(len(angles)):
            rotated1 = rotate_octahedral_exact(field1, angles[i:i+1], rotate_channels=True, mode="pairwise")
            rotated2 = rotate_octahedral_exact(field2, angles[i:i+1], rotate_channels=True, mode="pairwise")
            rotated_dot_sum = (rotated1 * rotated2).sum()

            assert torch.allclose(original_dot_sum, rotated_dot_sum, atol=1e-5), f"Rotation {i}"


# ============ Symmetric tensor field tests (C=6) ============

class TestSymmetricTensorField:

    def test_xx_tensor_rotation(self):
        """Test that T = diag(1,0,0) (xx only) rotates correctly."""
        field = torch.zeros(1, 6, 2, 2, 2)
        field[:, 0] = 1.0

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            angle = angles[i:i+1]
            R = matrices[i]

            T_in = tensor6_to_matrix(torch.tensor([1., 0., 0., 0., 0., 0.]))
            T_out = rotate_tensor(T_in, R)
            expected = matrix_to_tensor6(T_out)

            rotated = rotate_octahedral_exact(field, angle, rotate_channels=True, mode="pairwise")
            actual = rotated[0, :, 0, 0, 0]

            assert torch.allclose(actual, expected, atol=1e-5), f"Rotation {i}"

    def test_yy_tensor_rotation(self):
        """Test that T = diag(0,1,0) (yy only) rotates correctly."""
        field = torch.zeros(1, 6, 2, 2, 2)
        field[:, 1] = 1.0

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            angle = angles[i:i+1]
            R = matrices[i]

            T_in = tensor6_to_matrix(torch.tensor([0., 1., 0., 0., 0., 0.]))
            T_out = rotate_tensor(T_in, R)
            expected = matrix_to_tensor6(T_out)

            rotated = rotate_octahedral_exact(field, angle, rotate_channels=True, mode="pairwise")
            actual = rotated[0, :, 0, 0, 0]

            assert torch.allclose(actual, expected, atol=1e-5), f"Rotation {i}"

    def test_zz_tensor_rotation(self):
        """Test that T = diag(0,0,1) (zz only) rotates correctly."""
        field = torch.zeros(1, 6, 2, 2, 2)
        field[:, 2] = 1.0

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            angle = angles[i:i+1]
            R = matrices[i]

            T_in = tensor6_to_matrix(torch.tensor([0., 0., 1., 0., 0., 0.]))
            T_out = rotate_tensor(T_in, R)
            expected = matrix_to_tensor6(T_out)

            rotated = rotate_octahedral_exact(field, angle, rotate_channels=True, mode="pairwise")
            actual = rotated[0, :, 0, 0, 0]

            assert torch.allclose(actual, expected, atol=1e-5), f"Rotation {i}"

    def test_xy_tensor_rotation(self):
        """Test that off-diagonal xy component rotates correctly."""
        field = torch.zeros(1, 6, 2, 2, 2)
        field[:, 3] = 1.0

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            angle = angles[i:i+1]
            R = matrices[i]

            T_in = tensor6_to_matrix(torch.tensor([0., 0., 0., 1., 0., 0.]))
            T_out = rotate_tensor(T_in, R)
            expected = matrix_to_tensor6(T_out)

            rotated = rotate_octahedral_exact(field, angle, rotate_channels=True, mode="pairwise")
            actual = rotated[0, :, 0, 0, 0]

            assert torch.allclose(actual, expected, atol=1e-5), f"Rotation {i}"

    def test_identity_tensor_invariance(self):
        """Test that identity tensor T = diag(1,1,1) is invariant under all rotations."""
        field = torch.zeros(1, 6, 2, 2, 2)
        field[:, 0] = 1.0
        field[:, 1] = 1.0
        field[:, 2] = 1.0

        angles = get_all_octahedral_angles()
        expected = torch.tensor([1., 1., 1., 0., 0., 0.])

        for i in range(len(angles)):
            angle = angles[i:i+1]

            rotated = rotate_octahedral_exact(field, angle, rotate_channels=True, mode="pairwise")
            actual = rotated[0, :, 0, 0, 0]

            assert torch.allclose(actual, expected, atol=1e-5), f"Rotation {i}"

    def test_general_tensor_rotation(self):
        """Test a general symmetric tensor with all components nonzero."""
        field = torch.zeros(1, 6, 2, 2, 2)
        field[:, 0] = 1.0
        field[:, 1] = 2.0
        field[:, 2] = 3.0
        field[:, 3] = 0.5
        field[:, 4] = 0.3
        field[:, 5] = 0.2

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            angle = angles[i:i+1]
            R = matrices[i]

            T_in = tensor6_to_matrix(torch.tensor([1., 2., 3., 0.5, 0.3, 0.2]))
            T_out = rotate_tensor(T_in, R)
            expected = matrix_to_tensor6(T_out)

            rotated = rotate_octahedral_exact(field, angle, rotate_channels=True, mode="pairwise")
            actual = rotated[0, :, 0, 0, 0]

            assert torch.allclose(actual, expected, atol=1e-5), f"Rotation {i}"

    def test_identity_rotation(self):
        """Test that identity rotation leaves tensor field unchanged."""
        field = torch.zeros(1, 6, 2, 2, 2)
        field[0, 0, 0, 0, 0] = 1.0
        field[0, 3, 1, 0, 1] = 2.0

        angles = get_all_octahedral_angles()
        identity_angle = angles[0:1]

        rotated = rotate_octahedral_exact(field, identity_angle, rotate_channels=True, mode="pairwise")

        assert torch.allclose(rotated, field, atol=1e-5)

    def test_inverse_rotations(self):
        """Test that applying a rotation and its inverse returns the original tensor field."""
        field = torch.zeros(1, 6, 2, 2, 2)
        field[0, 0, 0, 0, 0] = 1.0
        field[0, 1, 0, 0, 0] = 2.0
        field[0, 3, 0, 0, 0] = 0.5

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            angle_i = angles[i:i+1]

            inv_idx = find_inverse_rotation(matrices, i)
            assert inv_idx is not None, f"Inverse of rotation {i} not found"

            rotated = rotate_octahedral_exact(field, angle_i, rotate_channels=True, mode="pairwise")
            restored = rotate_octahedral_exact(rotated, angles[inv_idx:inv_idx+1], rotate_channels=True, mode="pairwise")

            assert torch.allclose(restored, field, atol=1e-5), f"Rotation {i}"

    def test_rotation_composition(self):
        """Test that applying two rotations equals applying their composition."""
        field = torch.zeros(1, 6, 2, 2, 2)
        field[0, 0, 0, 0, 0] = 1.0
        field[0, 3, 0, 0, 0] = 0.5

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        i, j = 5, 11
        angle_i, angle_j = angles[i:i+1], angles[j:j+1]

        rotated_once = rotate_octahedral_exact(field, angle_i, rotate_channels=True, mode="pairwise")
        rotated_twice = rotate_octahedral_exact(rotated_once, angle_j, rotate_channels=True, mode="pairwise")

        composed_idx = find_composed_rotation(matrices, i, j)
        assert composed_idx is not None, "Composed rotation not found in group"

        rotated_composed = rotate_octahedral_exact(field, angles[composed_idx:composed_idx+1], rotate_channels=True, mode="pairwise")

        assert torch.allclose(rotated_twice, rotated_composed, atol=1e-5)


# ============ Symmetric tensor field stress tests (C=6) ============

class TestSymmetricTensorFieldStress:

    def test_random_field_inverse(self):
        """Test that R^{-1}(R(field)) = field for random symmetric tensor fields."""
        torch.manual_seed(47)
        field = torch.randn(1, 6, 4, 4, 4)

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            inv_idx = find_inverse_rotation(matrices, i)
            assert inv_idx is not None

            rotated = rotate_octahedral_exact(field, angles[i:i+1], rotate_channels=True, mode="pairwise")
            restored = rotate_octahedral_exact(rotated, angles[inv_idx:inv_idx+1], rotate_channels=True, mode="pairwise")

            assert torch.allclose(restored, field, atol=1e-5), f"Rotation {i}"

    def test_random_field_composition(self):
        """Test that R_j(R_i(field)) = R_{ji}(field) for random symmetric tensor fields."""
        torch.manual_seed(48)
        field = torch.randn(1, 6, 4, 4, 4)

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(0, len(angles), 4):
            for j in range(0, len(angles), 4):
                composed_idx = find_composed_rotation(matrices, i, j)
                assert composed_idx is not None

                rotated_twice = rotate_octahedral_exact(
                    rotate_octahedral_exact(field, angles[i:i+1], rotate_channels=True, mode="pairwise"),
                    angles[j:j+1], rotate_channels=True, mode="pairwise"
                )
                rotated_composed = rotate_octahedral_exact(field, angles[composed_idx:composed_idx+1], rotate_channels=True, mode="pairwise")

                assert torch.allclose(rotated_twice, rotated_composed, atol=1e-5), f"Rotations {i}, {j}"

    def test_random_field_norm_preservation(self):
        """Test that rotation preserves the L2 norm of the field."""
        torch.manual_seed(49)
        field = torch.randn(1, 6, 4, 4, 4)
        original_norm = torch.norm(field)

        angles = get_all_octahedral_angles()

        for i in range(len(angles)):
            rotated = rotate_octahedral_exact(field, angles[i:i+1], rotate_channels=True, mode="pairwise")
            rotated_norm = torch.norm(rotated)

            assert torch.allclose(original_norm, rotated_norm, atol=1e-5), f"Rotation {i}"

    def test_random_field_trace_preservation(self):
        """Test that rotation preserves the sum of traces over the field."""
        torch.manual_seed(50)
        field = torch.randn(1, 6, 4, 4, 4)
        original_trace_sum = (field[:, 0] + field[:, 1] + field[:, 2]).sum()

        angles = get_all_octahedral_angles()

        for i in range(len(angles)):
            rotated = rotate_octahedral_exact(field, angles[i:i+1], rotate_channels=True, mode="pairwise")
            rotated_trace_sum = (rotated[:, 0] + rotated[:, 1] + rotated[:, 2]).sum()

            assert torch.allclose(original_trace_sum, rotated_trace_sum, atol=1e-5), f"Rotation {i}"

    def test_random_field_frobenius_norm_preservation(self):
        """Test that rotation preserves the sum of Frobenius norms over the field."""
        torch.manual_seed(51)
        field = torch.randn(1, 6, 4, 4, 4)

        original_frob_sum = (
            field[:, 0]**2 + field[:, 1]**2 + field[:, 2]**2 +
            2 * (field[:, 3]**2 + field[:, 4]**2 + field[:, 5]**2)
        ).sum()

        angles = get_all_octahedral_angles()

        for i in range(len(angles)):
            rotated = rotate_octahedral_exact(field, angles[i:i+1], rotate_channels=True, mode="pairwise")
            rotated_frob_sum = (
                rotated[:, 0]**2 + rotated[:, 1]**2 + rotated[:, 2]**2 +
                2 * (rotated[:, 3]**2 + rotated[:, 4]**2 + rotated[:, 5]**2)
            ).sum()

            assert torch.allclose(original_frob_sum, rotated_frob_sum, atol=1e-5), f"Rotation {i}"

    def test_random_field_determinant_preservation(self):
        """Test that rotation preserves the determinant of each tensor."""
        torch.manual_seed(52)
        field = torch.randn(1, 6, 2, 2, 2)

        angles = get_all_octahedral_angles()

        original_dets = []
        for d in range(2):
            for h in range(2):
                for w in range(2):
                    T = tensor6_to_matrix(field[0, :, d, h, w])
                    original_dets.append(torch.det(T))
        original_det_sum = torch.tensor(original_dets).sum()

        for i in range(len(angles)):
            rotated = rotate_octahedral_exact(field, angles[i:i+1], rotate_channels=True, mode="pairwise")

            rotated_dets = []
            for d in range(2):
                for h in range(2):
                    for w in range(2):
                        T = tensor6_to_matrix(rotated[0, :, d, h, w])
                        rotated_dets.append(torch.det(T))
            rotated_det_sum = torch.tensor(rotated_dets).sum()

            assert torch.allclose(original_det_sum, rotated_det_sum, atol=1e-4), f"Rotation {i}"

    def test_random_field_eigenvalue_preservation(self):
        """Test that rotation preserves sum of eigenvalues over the field."""
        torch.manual_seed(53)
        field = torch.randn(1, 6, 2, 2, 2)

        angles = get_all_octahedral_angles()

        original_eig_sum = torch.tensor(0.0)
        for d in range(2):
            for h in range(2):
                for w in range(2):
                    T = tensor6_to_matrix(field[0, :, d, h, w])
                    eigs = torch.linalg.eigvalsh(T)
                    original_eig_sum += eigs.sum()

        for i in range(len(angles)):
            rotated = rotate_octahedral_exact(field, angles[i:i+1], rotate_channels=True, mode="pairwise")

            rotated_eig_sum = torch.tensor(0.0)
            for d in range(2):
                for h in range(2):
                    for w in range(2):
                        T = tensor6_to_matrix(rotated[0, :, d, h, w])
                        eigs = torch.linalg.eigvalsh(T)
                        rotated_eig_sum += eigs.sum()

            assert torch.allclose(original_eig_sum, rotated_eig_sum, atol=1e-5), f"Rotation {i}"


# ============ Full tensor field tests (C=9) ============

class TestFullTensorField:

    def test_xx_tensor_rotation(self):
        """Test that T with only (0,0) component rotates correctly."""
        field = torch.zeros(1, 9, 2, 2, 2)
        field[:, 0] = 1.0

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            angle = angles[i:i+1]
            R = matrices[i]

            T_in = torch.zeros(3, 3)
            T_in[0, 0] = 1.0
            T_out = rotate_tensor(T_in, R)
            expected = matrix_to_tensor9(T_out)

            rotated = rotate_octahedral_exact(field, angle, rotate_channels=True, mode="pairwise")
            actual = rotated[0, :, 0, 0, 0]

            assert torch.allclose(actual, expected, atol=1e-5), f"Rotation {i}"

    def test_xy_tensor_rotation(self):
        """Test that T with only (0,1) component rotates correctly."""
        field = torch.zeros(1, 9, 2, 2, 2)
        field[:, 1] = 1.0

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            angle = angles[i:i+1]
            R = matrices[i]

            T_in = torch.zeros(3, 3)
            T_in[0, 1] = 1.0
            T_out = rotate_tensor(T_in, R)
            expected = matrix_to_tensor9(T_out)

            rotated = rotate_octahedral_exact(field, angle, rotate_channels=True, mode="pairwise")
            actual = rotated[0, :, 0, 0, 0]

            assert torch.allclose(actual, expected, atol=1e-5), f"Rotation {i}"

    def test_yx_tensor_rotation(self):
        """Test that T with only (1,0) component rotates correctly (asymmetric)."""
        field = torch.zeros(1, 9, 2, 2, 2)
        field[:, 3] = 1.0

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            angle = angles[i:i+1]
            R = matrices[i]

            T_in = torch.zeros(3, 3)
            T_in[1, 0] = 1.0
            T_out = rotate_tensor(T_in, R)
            expected = matrix_to_tensor9(T_out)

            rotated = rotate_octahedral_exact(field, angle, rotate_channels=True, mode="pairwise")
            actual = rotated[0, :, 0, 0, 0]

            assert torch.allclose(actual, expected, atol=1e-5), f"Rotation {i}"

    def test_asymmetric_tensor_rotation(self):
        """Test an asymmetric tensor where T[0,1] != T[1,0]."""
        field = torch.zeros(1, 9, 2, 2, 2)
        field[:, 1] = 1.0
        field[:, 3] = 2.0

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            angle = angles[i:i+1]
            R = matrices[i]

            T_in = torch.zeros(3, 3)
            T_in[0, 1] = 1.0
            T_in[1, 0] = 2.0
            T_out = rotate_tensor(T_in, R)
            expected = matrix_to_tensor9(T_out)

            rotated = rotate_octahedral_exact(field, angle, rotate_channels=True, mode="pairwise")
            actual = rotated[0, :, 0, 0, 0]

            assert torch.allclose(actual, expected, atol=1e-5), f"Rotation {i}"

    def test_identity_tensor_invariance(self):
        """Test that identity tensor T = I is invariant under all rotations."""
        field = torch.zeros(1, 9, 2, 2, 2)
        field[:, 0] = 1.0
        field[:, 4] = 1.0
        field[:, 8] = 1.0

        angles = get_all_octahedral_angles()
        expected = matrix_to_tensor9(torch.eye(3))

        for i in range(len(angles)):
            angle = angles[i:i+1]

            rotated = rotate_octahedral_exact(field, angle, rotate_channels=True, mode="pairwise")
            actual = rotated[0, :, 0, 0, 0]

            assert torch.allclose(actual, expected, atol=1e-5), f"Rotation {i}"

    def test_general_tensor_rotation(self):
        """Test a general non-symmetric tensor with all components nonzero."""
        T_in = torch.tensor([
            [1.0, 0.5, 0.3],
            [0.2, 2.0, 0.4],
            [0.1, 0.6, 3.0]
        ])

        field = torch.zeros(1, 9, 2, 2, 2)
        field[0, :, :, :, :] = matrix_to_tensor9(T_in).view(9, 1, 1, 1)

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            angle = angles[i:i+1]
            R = matrices[i]

            T_out = rotate_tensor(T_in, R)
            expected = matrix_to_tensor9(T_out)

            rotated = rotate_octahedral_exact(field, angle, rotate_channels=True, mode="pairwise")
            actual = rotated[0, :, 0, 0, 0]

            assert torch.allclose(actual, expected, atol=1e-5), f"Rotation {i}"

    def test_identity_rotation(self):
        """Test that identity rotation leaves tensor field unchanged."""
        field = torch.zeros(1, 9, 2, 2, 2)
        field[0, 0, 0, 0, 0] = 1.0
        field[0, 1, 1, 0, 1] = 2.0
        field[0, 3, 0, 1, 0] = 3.0

        angles = get_all_octahedral_angles()
        identity_angle = angles[0:1]

        rotated = rotate_octahedral_exact(field, identity_angle, rotate_channels=True, mode="pairwise")

        assert torch.allclose(rotated, field, atol=1e-5)

    def test_inverse_rotations(self):
        """Test that applying a rotation and its inverse returns the original tensor field."""
        T_in = torch.tensor([
            [1.0, 0.5, 0.3],
            [0.2, 2.0, 0.4],
            [0.1, 0.6, 3.0]
        ])

        field = torch.zeros(1, 9, 2, 2, 2)
        field[0, :, 0, 0, 0] = matrix_to_tensor9(T_in)

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            angle_i = angles[i:i+1]

            inv_idx = find_inverse_rotation(matrices, i)
            assert inv_idx is not None, f"Inverse of rotation {i} not found"

            rotated = rotate_octahedral_exact(field, angle_i, rotate_channels=True, mode="pairwise")
            restored = rotate_octahedral_exact(rotated, angles[inv_idx:inv_idx+1], rotate_channels=True, mode="pairwise")

            assert torch.allclose(restored, field, atol=1e-5), f"Rotation {i}"

    def test_rotation_composition(self):
        """Test that applying two rotations equals applying their composition."""
        T_in = torch.tensor([
            [1.0, 0.5, 0.3],
            [0.2, 2.0, 0.4],
            [0.1, 0.6, 3.0]
        ])

        field = torch.zeros(1, 9, 2, 2, 2)
        field[0, :, 0, 0, 0] = matrix_to_tensor9(T_in)

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        i, j = 5, 11
        angle_i, angle_j = angles[i:i+1], angles[j:j+1]

        rotated_once = rotate_octahedral_exact(field, angle_i, rotate_channels=True, mode="pairwise")
        rotated_twice = rotate_octahedral_exact(rotated_once, angle_j, rotate_channels=True, mode="pairwise")

        composed_idx = find_composed_rotation(matrices, i, j)
        assert composed_idx is not None, "Composed rotation not found in group"

        rotated_composed = rotate_octahedral_exact(field, angles[composed_idx:composed_idx+1], rotate_channels=True, mode="pairwise")

        assert torch.allclose(rotated_twice, rotated_composed, atol=1e-5)


# ============ Full tensor field stress tests (C=9) ============

class TestFullTensorFieldStress:

    def test_random_field_inverse(self):
        """Test that R^{-1}(R(field)) = field for random full tensor fields."""
        torch.manual_seed(54)
        field = torch.randn(1, 9, 4, 4, 4)

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            inv_idx = find_inverse_rotation(matrices, i)
            assert inv_idx is not None

            rotated = rotate_octahedral_exact(field, angles[i:i+1], rotate_channels=True, mode="pairwise")
            restored = rotate_octahedral_exact(rotated, angles[inv_idx:inv_idx+1], rotate_channels=True, mode="pairwise")

            assert torch.allclose(restored, field, atol=1e-5), f"Rotation {i}"

    def test_random_field_composition(self):
        """Test that R_j(R_i(field)) = R_{ji}(field) for random full tensor fields."""
        torch.manual_seed(55)
        field = torch.randn(1, 9, 4, 4, 4)

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(0, len(angles), 4):
            for j in range(0, len(angles), 4):
                composed_idx = find_composed_rotation(matrices, i, j)
                assert composed_idx is not None

                rotated_twice = rotate_octahedral_exact(
                    rotate_octahedral_exact(field, angles[i:i+1], rotate_channels=True, mode="pairwise"),
                    angles[j:j+1], rotate_channels=True, mode="pairwise"
                )
                rotated_composed = rotate_octahedral_exact(field, angles[composed_idx:composed_idx+1], rotate_channels=True, mode="pairwise")

                assert torch.allclose(rotated_twice, rotated_composed, atol=1e-5), f"Rotations {i}, {j}"

    def test_random_field_norm_preservation(self):
        """Test that rotation preserves the L2 norm of the field."""
        torch.manual_seed(56)
        field = torch.randn(1, 9, 4, 4, 4)
        original_norm = torch.norm(field)

        angles = get_all_octahedral_angles()

        for i in range(len(angles)):
            rotated = rotate_octahedral_exact(field, angles[i:i+1], rotate_channels=True, mode="pairwise")
            rotated_norm = torch.norm(rotated)

            assert torch.allclose(original_norm, rotated_norm, atol=1e-5), f"Rotation {i}"

    def test_random_field_trace_preservation(self):
        """Test that rotation preserves the sum of traces over the field."""
        torch.manual_seed(57)
        field = torch.randn(1, 9, 4, 4, 4)
        original_trace_sum = (field[:, 0] + field[:, 4] + field[:, 8]).sum()

        angles = get_all_octahedral_angles()

        for i in range(len(angles)):
            rotated = rotate_octahedral_exact(field, angles[i:i+1], rotate_channels=True, mode="pairwise")
            rotated_trace_sum = (rotated[:, 0] + rotated[:, 4] + rotated[:, 8]).sum()

            assert torch.allclose(original_trace_sum, rotated_trace_sum, atol=1e-5), f"Rotation {i}"

    def test_random_field_frobenius_norm_preservation(self):
        """Test that rotation preserves the sum of Frobenius norms over the field."""
        torch.manual_seed(58)
        field = torch.randn(1, 9, 4, 4, 4)
        original_frob_sum = (field ** 2).sum()

        angles = get_all_octahedral_angles()

        for i in range(len(angles)):
            rotated = rotate_octahedral_exact(field, angles[i:i+1], rotate_channels=True, mode="pairwise")
            rotated_frob_sum = (rotated ** 2).sum()

            assert torch.allclose(original_frob_sum, rotated_frob_sum, atol=1e-5), f"Rotation {i}"

    def test_random_field_determinant_preservation(self):
        """Test that rotation preserves sum of determinants over the field."""
        torch.manual_seed(59)
        field = torch.randn(1, 9, 2, 2, 2)

        angles = get_all_octahedral_angles()

        original_det_sum = torch.tensor(0.0)
        for d in range(2):
            for h in range(2):
                for w in range(2):
                    T = tensor9_to_matrix(field[0, :, d, h, w])
                    original_det_sum += torch.det(T)

        for i in range(len(angles)):
            rotated = rotate_octahedral_exact(field, angles[i:i+1], rotate_channels=True, mode="pairwise")

            rotated_det_sum = torch.tensor(0.0)
            for d in range(2):
                for h in range(2):
                    for w in range(2):
                        T = tensor9_to_matrix(rotated[0, :, d, h, w])
                        rotated_det_sum += torch.det(T)

            assert torch.allclose(original_det_sum, rotated_det_sum, atol=1e-4), f"Rotation {i}"

    def test_random_field_eigenvalue_preservation(self):
        """Test that rotation preserves sum of eigenvalue magnitudes over the field."""
        torch.manual_seed(60)
        field = torch.randn(1, 9, 2, 2, 2)

        angles = get_all_octahedral_angles()

        original_eig_sum = torch.tensor(0.0)
        for d in range(2):
            for h in range(2):
                for w in range(2):
                    T = tensor9_to_matrix(field[0, :, d, h, w])
                    eigs = torch.linalg.eigvals(T).abs()
                    original_eig_sum += eigs.sum()

        for i in range(len(angles)):
            rotated = rotate_octahedral_exact(field, angles[i:i+1], rotate_channels=True, mode="pairwise")

            rotated_eig_sum = torch.tensor(0.0)
            for d in range(2):
                for h in range(2):
                    for w in range(2):
                        T = tensor9_to_matrix(rotated[0, :, d, h, w])
                        eigs = torch.linalg.eigvals(T).abs()
                        rotated_eig_sum += eigs.sum()

            assert torch.allclose(original_eig_sum, rotated_eig_sum, atol=1e-4), f"Rotation {i}"

    def test_random_field_antisymmetric_part_norm(self):
        """Test that rotation preserves sum of antisymmetric part norms over the field."""
        torch.manual_seed(62)
        field = torch.randn(1, 9, 4, 4, 4)

        original_antisym_sum = (
            (field[:, 1] - field[:, 3])**2 +
            (field[:, 2] - field[:, 6])**2 +
            (field[:, 5] - field[:, 7])**2
        ).sum() / 2

        angles = get_all_octahedral_angles()

        for i in range(len(angles)):
            rotated = rotate_octahedral_exact(field, angles[i:i+1], rotate_channels=True, mode="pairwise")
            rotated_antisym_sum = (
                (rotated[:, 1] - rotated[:, 3])**2 +
                (rotated[:, 2] - rotated[:, 6])**2 +
                (rotated[:, 5] - rotated[:, 7])**2
            ).sum() / 2

            assert torch.allclose(original_antisym_sum, rotated_antisym_sum, atol=1e-5), f"Rotation {i}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])