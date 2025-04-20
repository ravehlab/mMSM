import numpy as np
import numba
from mmsm.mmsm_base.base_discretizer import BaseDiscretizer
from mmsm.mmsm_base.proc.kcenters import KCentersDiscretizer


@numba.njit
def dihedral_angle(p0, p1, p2, p3):
    """Dihedral angle formula, adapted from
    https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python"""
    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= np.linalg.norm(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))


def disc_dhdrl(traj):
    phis = [dihedral_angle(coords[4], coords[6], coords[8], coords[14]) for coords in traj]
    psis = [dihedral_angle(coords[6], coords[8], coords[14], coords[16]) for coords in traj]
    return phis, psis


def disc_dhdrl_vs(traj):
    """Use this when working with velocities"""
    phis = [dihedral_angle(coords[0][4], coords[0][6], coords[0][8], coords[0][14]) for coords in traj]
    psis = [dihedral_angle(coords[0][6], coords[0][8], coords[0][14], coords[0][16]) for coords in traj]
    return phis, psis

def disc_dhdrl_7(traj):
    dhdrls = [[
                dihedral_angle(coords[0][1], coords[0][4], coords[0][6], coords[0][8]),  # omega2
                dihedral_angle(coords[0][4], coords[0][6], coords[0][8], coords[0][14]),  # phi
                dihedral_angle(coords[0][6], coords[0][8], coords[0][14], coords[0][16]),  # psi
                dihedral_angle(coords[0][8], coords[0][14], coords[0][16], coords[0][18]),  # omega1

                dihedral_angle(coords[0][0], coords[0][1], coords[0][4], coords[0][6]),  # chi2
                dihedral_angle(coords[0][19], coords[0][18], coords[0][16], coords[0][14]),  # chi1
                dihedral_angle(coords[0][6], coords[0][8], coords[0][10], coords[0][11])  # chi3
                ] for coords in traj]
    return dhdrls


class DialanineDiscretizer(BaseDiscretizer):
    def __init__(self, cutoff, representative_sample_size=10):
        super().__init__(representative_sample_size)
        self._kcenters = KCentersDiscretizer(cutoff, representative_sample_size)

    @property
    def n_states(self):
        return self._kcenters.n_states

    def _coarse_grain_states(self, data):
        # 0 - Phi, 1 - Psi
        phis_psis = np.vstack((disc_dhdrl(data))).T
        return self._kcenters._coarse_grain_states(phis_psis)

    def get_centers_by_ids(self, cluster_ids):
        return self._kcenters.get_centers_by_ids(cluster_ids)


class DialanineDiscretizerV(BaseDiscretizer):
    """Used when trajectories contain velocities."""
    def __init__(self, cutoff, representative_sample_size=10):
        super().__init__(representative_sample_size)
        self._kcenters = KCentersDiscretizer(cutoff, representative_sample_size)

    @property
    def n_states(self):
        return self._kcenters.n_states

    def _coarse_grain_states(self, data):
        # 0 - Phi, 1 - Psi
        phis_psis = np.vstack((disc_dhdrl_vs(data))).T
        return self._kcenters._coarse_grain_states(phis_psis)

    def get_centers_by_ids(self, cluster_ids):
        return self._kcenters.get_centers_by_ids(cluster_ids)


class AlanineAngleDiscretizer(BaseDiscretizer):
    def __init__(self, cutoff, representative_sample_size=10):
        super().__init__(representative_sample_size)
        self._kcenters = KCentersDiscretizer(cutoff, representative_sample_size)

    @property
    def n_states(self):
        return self._kcenters.n_states

    def _coarse_grain_states(self, data):
        # 0 - Phi, 1 - Psi
        angles = np.vstack(disc_dhdrl_7(data))
        # phis_psis = np.vstack((disc_dhdrl_vs(data))).T
        return self._kcenters._coarse_grain_states(angles)

    def get_centers_by_ids(self, cluster_ids):
        return self._kcenters.get_centers_by_ids(cluster_ids)
