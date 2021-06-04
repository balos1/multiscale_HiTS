import torch
import numpy as np
from pymgrit.core.application import Application
from pymgrit.core.vector import Vector

class VectorHits(Vector):
    """
    Vector class for use with HiTS
    """

    def __init__(self, dims=None, data=None):
        super().__init__()

        self.dims = dims
        self.data = None

        if data is not None:
          if isinstance(data, torch.Tensor):
            self.data = data.clone().detach()
            self.dims = data.shape
          else:
            self.data = torch.tensor(data, dtype=torch.float32)
            self.dims = data.shape
        elif dims is not None:
          self.data = torch.zeros(*dims)
          self.dims = dims

    def set_values(self, data):
        self.data = data.clone().detach()

    def get_values(self):
        return self.data

    def clone(self):
        return VectorHits(data=self.data)

    def clone_zero(self):
        return VectorHits(dims=self.data.shape)

    def clone_rand(self):
        return VectorHits(data=torch.rand(self.data.shape))

    def __add__(self, other):
        tmp = self.clone_zero()
        tmp.set_values(self.get_values() + other.get_values())
        return tmp

    def __sub__(self, other):
        tmp = self.clone_zero()
        tmp.set_values(self.get_values() - other.get_values())
        return tmp

    def __mul__(self, other):
        tmp = self.clone_zero()
        tmp.set_values(self.get_values() * other)
        return tmp

    def norm(self):
        return self.data.norm()

    def pack(self):
        return self.data

    def unpack(self, tensor):
        self.set_values(tensor)

    def __repr__(self):
      return np.array_repr(self.data.detach().numpy())


class MgritHits(Application):
    """
    """

    def __init__(self, ic, stepfn=None, model=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # neural net stepper
        self.model = model

        # Set the data structure for any user-defined time point
        self.vector_template = VectorHits(dims=ic.shape)

        # Set the initial condition
        self.vector_t_start = VectorHits(data=ic)

        self.stepfn = stepfn

        self.history = []

    # Time integration routine
    def step(self, u_start: VectorHits, t_start: float, t_stop: float) -> VectorHits:
        init = u_start.get_values()
        if self.model:
          u_n = self.model.forward(u_start.get_values())
          return VectorHits(data=u_n)
        else:
          # u_n = problem.step_backward_euler(u_start.get_values().numpy(), t_start, t_stop)
          u_n = self.stepfn(u_start.get_values().numpy(), t_start, t_stop)
          return VectorHits(data=u_n)
