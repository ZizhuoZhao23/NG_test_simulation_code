__copyright__ = "Copyright (C) 2019 Zachary J Weiner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


import numpy as np
from pystella import DynamicField, Field
from pystella.field import diff
from pymbolic import var

__doc__ = """
.. currentmodule:: pystella
.. autoclass:: Sector
.. autoclass:: ScalarSector
.. autoclass:: TensorPerturbationSector

.. currentmodule:: pystella.sectors
.. autofunction:: get_rho_and_p
"""

eta = [-1, 1, 1, 1]


class Sector:
    """
    A unimplemented base class defining the methods and properties needed for
    code generation for, e.g., preheating simulations.

    .. automethod:: __init__
    .. autoattribute:: rhs_dict
    .. autoattribute:: reducers
    .. automethod:: stress_tensor
    """

    def __init__(self):
        """
        Processes input needed to specify a model for the particular
        :class:`Sector`.
        """

        raise NotImplementedError

    @property
    def rhs_dict(self):
        """
        An ``@property`` method returning a :class:`dict` specifying the system
        of equations to be time-integrated.
        See the documentation of :class:`~pystella.step.Stepper`.
        """
        raise NotImplementedError

    @property
    def reducers(self):
        """
        An ``@property`` method returning :class:`dict` specifying the quantities
        to be computed, e.g., energy components for :class:`Expansion` and output.
        See the documentation of :class:`Reduction`.
        """
        raise NotImplementedError

    def stress_tensor(self, mu, nu, drop_trace=True):
        """
        :arg drop_trace: Whether to drop the term
            :math:`g_{\\mu\\nu} \\mathcal{L}`.
            Defauls to *False*.

        :returns: The component :math:`T_{\\mu\\nu}` of the stress-energy
            tensor of the particular :class:`Sector`.
            Used by :class:`TensorPerturbationSector`, with ``drop_trace=True``.
        """
        raise NotImplementedError


class ScalarSector(Sector):
    """
    A :class:`Sector` of scalar fields.

    :arg nscalars: The total number of scalar fields.

    The following keyword-only arguments are recognized:

    :arg f: The :class:`DynamicField` of scalar fields.
        Defaults to ``DynamicField("f", offset="h", shape=(nscalars,))``.

    :arg potential: A :class:`~collections.abc.Callable` which takes as input a
        :mod:`pymbolic` expression or a :class:`list` thereof, returning
        the potential of the scalar fields.
        Defaults to ``lambda x: 0``.

    :raises ValueError: if a particular field is coupled to its own kinetic
        term.
    """

    def __init__(self, nscalars, **kwargs):
        self.nscalars = nscalars
        self.f = kwargs.pop("f", DynamicField("f", offset="h", shape=(nscalars,)))
        self.E = kwargs.pop("E", DynamicField("E", offset="h", shape=(3,)))
        self.potential = kwargs.pop("potential", lambda x: 0)

    @property
    def rhs_dict(self):
        f, E = self.f, self.E
        scalar1, scalar2 = DynamicField("scalar1", offset="h", shape=(2,)), DynamicField("scalar2", offset="h", shape=(2,))
        link1, link2 = DynamicField("link1", offset="h", shape=(3,)), DynamicField("link2", offset="h", shape=(3,))
        H = Field("hubble", indices=[])
        a = Field("a", indices=[])
        
        rhs_dict = {}
        V = self.potential(f)
    
        for fld in range(self.nscalars):
            rhs_dict[f[fld]] = f.dot[fld]
            rhs_dict[f.dot[fld]] = scalar1[fld] - scalar2[fld]
        
        for d in range(3):
            rhs_dict[E[d]] = link1[d] + link2[d]
        
        return rhs_dict
        
        
    @property
    def reducers(self):
        f = self.f
        a = var("a")
        
        reducers = {}
        reducers["kinetic"] = [f.dot[fld]**2 / 2 / a**2
                               for fld in range(self.nscalars)]
        reducers["potential"] = [self.potential(f)]
        reducers["gradient"] = [- f[fld] * f.lap[fld] / 2 / a**2 
                                for fld in range(self.nscalars)]
        return reducers

            
    def stress_tensor(self, mu, nu, n, drop_trace=False):
        f = self.f
        a = Field("a", indices=[])
        
        Tmunu = f.d(n, mu) * f.d(n, nu)

        if drop_trace:
            return Tmunu
        else:
            metric = np.diag((-1/a**2, 1/a**2, 1/a**2, 1/a**2))  # contravariant
            lag = (- sum(sum(metric[mu, nu] * f.d(fld, mu) * f.d(fld, nu)
                             for mu in range(4) for nu in range(4))
                         for fld in range(0,1)) / 2
                   - self.potential(self.f))
            metric = np.diag((-a**2, a**2, a**2, a**2))  # covariant
            return Tmunu + metric[mu, nu] * lag

def tensor_index(i, j):
    a = i if i <= j else j
    b = j if i <= j else i
    return (7 - a) * a // 2 - 4 + b


class TensorPerturbationSector:
    """
    A :class:`Sector` of tensor perturbations.

    :arg sectors: The :class:`Sector`\\ s whose :meth:`~Sector.stress_tensor`\\ s
        source the tensor perturbations.

    The following keyword-only arguments are recognized:

    :arg hij: The :class:`DynamicField` of tensor fields.
        Defaults to ``DynamicField("hij", offset="h", shape=(6,))``.
    """

    def __init__(self, nscalars, **kwargs):
        self.hijs = kwargs.pop("hijs", DynamicField("hijs", offset="h", shape=(6,)))
        self.hijg = kwargs.pop("hijg", DynamicField("hijg", offset="h", shape=(6,)))
        self.nscalars = nscalars

    @property
    def rhs_dict(self):
        hijs = self.hijs
        hijg = self.hijg
        H = Field("hubble", indices=[])

        scalar = DynamicField("scalar", shape=(6,))
        gauge = DynamicField("gauge", shape=(6,))

        rhs_dict = {}

        for i in range(1, 4):
            for j in range(i, 4):
                fld = tensor_index(i, j)

                rhs_dict[hijs[fld]] = hijs.dot[fld]
                rhs_dict[hijs.dot[fld]] = (hijs.lap[fld] - 2*H*hijs.dot[fld] + 16 * np.pi * scalar[fld])

                rhs_dict[hijg[fld]] = hijg.dot[fld]
                rhs_dict[hijg.dot[fld]] = (hijg.lap[fld] - 2*H*hijg.dot[fld] + 16 * np.pi * gauge[fld])

        return rhs_dict

    @property
    def reducers(self):
        return {}


def get_rho_and_p(energy):
    """
    Convenience callback for energy reductions which computes :math:`\\rho` and
    :math:`P`.

    :arg energy: A dictionary of energy components as returned by
        :class:`~pystella.Reduction`.
    """

    energy["total"] = sum(sum(e) for e in energy.values())
    energy["pressure"] = 0
    
    if "kinetic" in energy:
        energy["pressure"] += sum(energy["kinetic"])
    if "gradient" in energy:
        energy["pressure"] += - sum(energy["gradient"]) / 3
    if "potential" in energy:
        energy["pressure"] += - sum(energy["potential"])
    if "radiation" in energy:
        energy["pressure"] += sum(energy["radiation"]) / 3
    return energy
