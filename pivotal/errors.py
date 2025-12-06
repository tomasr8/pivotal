class PivotalError(Exception):
    pass


class Unbounded(PivotalError):
    pass


class Infeasible(PivotalError):
    pass


class AbsoluteValueRequiresMILP(PivotalError):
    pass
