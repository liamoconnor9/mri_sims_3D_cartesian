from dedalus.core import SpectralOperator1D
from dedalus.tools.dispatch import MultiClass

class Undiffuse(SpectralOperator1D, metaclass=MultiClass):
    """
    Differentiation along one dimension.

    Parameters
    ----------
    operand : number or Operand object
    space : Space object

    """

    name = "Diff"

    def __init__(self, operand, coord, out=None):
        super().__init__(operand, out=out)
        # SpectralOperator requirements
        self.coord = coord
        self.input_basis = operand.domain.get_basis(coord)
        self.output_basis = self._output_basis(self.input_basis)
        self.first_axis = coord.axis
        self.last_axis = coord.axis
        self.axis = coord.axis
        # LinearOperator requirements
        self.operand = operand
        # FutureField requirements
        self.domain = operand.domain.substitute_basis(self.input_basis, self.output_basis)
        self.tensorsig = operand.tensorsig
        self.dtype = operand.dtype

    @classmethod
    def _check_args(cls, operand, coord, out=None):
        # Dispatch by operand basis
        if isinstance(operand, Operand):
            basis = operand.domain.get_basis(coord)
            if isinstance(basis, cls.input_basis_type):
                return True
        return False

    def new_operand(self, operand, **kw):
        return Differentiate(operand, self.coord, **kw)

    @staticmethod
    def _output_basis(input_basis):
        # Subclasses must implement
        raise NotImplementedError()

    def __str__(self):
        return 'd{!s}({!s})'.format(self.coord.name, self.operand)

    def _expand_multiply(self, operand, vars):
        """Expand over multiplication."""
        args = operand.args
        # Apply product rule to factors
        partial_diff = lambda i: prod([self.new_operand(arg) if i==j else arg for j,arg in enumerate(args)])
        return sum((partial_diff(i) for i in range(len(args))))
