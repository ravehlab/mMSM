import IMP.atom
import IMP.algebra
import IMP.rmf
import IMP.core
import IMP.container
import IMP.display
import IMP.npctransport
import numpy as np

class CustomDistanceScore1(IMP.UnaryFunction):
    def __init__(self):
        IMP.UnaryFunction.__init__(self)

    def evaluate_with_derivative(self, d):
        return self.evaluate(d),0.0000000000045 * (2*((d-34)**2-1/25)*((d-27)**2-1/20)*(d-221/4)**2*(d-51)**2\
               *(d-44)**2*(d-91/4)+2*((d-34)**2-1/25)*((d-91/4)**2-3/50)*(d-221/4)**2*\
               (d-51)**2*(d-44)**2*(d-27)+2*((d-27)**2-1/20)*((d-91/4)**2-3/50)*\
               (d-221/4)**2*(d-51)**2*(d-44)**2*(d-34)+2*((d-34)**2-1/25)*\
               ((d-27)**2-1/20)*((d-91/4)**2-3/50)*(d-221/4)**2*(d-51)**2*\
               (d-44)+2*((d-34)**2-1/25)*((d-27)**2-1/20)*((d-91/4)**2-3/50)*\
               (d-221/4)**2*(d-51)*(d-44)**2+2*((d-34)**2-1/25)*((d-27)**2-1/20)*\
               ((d-91/4)**2-3/50)*(d-221/4)*(d-51)**2*(d-44)**2 ) + (48*np.cos(8*d))/5

    def evaluate(self, d):
        return 0.0000000000045 * ((d - 22.75)**2 - 0.06) * ((d - 27)**2 - 0.05) * ((d - 34)**2 - 0.04) \
               * (d - 44)**2 * (d - 51)**2 * (d - 55.25)**2 + 1.2 * np.sin(8*d) + 2


def create_beads(num_beads, model, model_h, r=5.0, m=1.0):
    beads = []
    for i in range(num_beads):
        p = IMP.Particle(model, "p{0}".format(i + 1))
        p_as_xyzr = IMP.core.XYZR.setup_particle(p)
        p_as_xyzr.set_coordinates_are_optimized(True)
        p_as_xyzr.set_radius(r)
        IMP.atom.Mass.setup_particle(p, m)
        IMP.atom.Diffusion.setup_particle(p)
        IMP.atom.Hierarchy.setup_particle(p)
        IMP.display.Colored.setup_particle(p, IMP.display.get_display_color(0))
        beads.append(p)
        model_h.add_child(p)
    return beads


def get_model_2_spheres(score_fn, rs=False):
    """2 spheres model, with a restraint on their distance, given by score_fn."""
    m = IMP.Model()
    p_root = IMP.Particle(m, "root")
    h_root = IMP.atom.Hierarchy.setup_particle(p_root)
    beads = create_beads(2, m, h_root)
    dps = IMP.core.DistancePairScore(score_fn())
    pr = IMP.core.PairRestraint(m, dps, beads)
    evr = IMP.core.ExcludedVolumeRestraint(beads, 1.0, 10.0)
    rsf = IMP.core.RestraintsScoringFunction([pr, evr])
    if rs:
        return m, rsf, [pr, evr]
    return m, rsf


def model1():
    """2 sphere model with dual basin energy function."""
    return get_model_2_spheres(CustomDistanceScore1)
