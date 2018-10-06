import functools
import itertools
from fglib import graphs, nodes, rv, inference


def get_variable_dimensions(factor_graph: graphs.FactorGraph):
    variable_dimensions = {}
    for factor in factor_graph.get_fnodes():
        var_shapes = zip(factor.factor.dim, factor.factor.pmf.shape)
        for (variable, shape) in var_shapes:
            if ((variable in variable_dimensions) and
                    (variable_dimensions[variable] != shape)):
                raise ValueError(
                    "Variable {} has inconsistent shape!"
                    .format(variable.__str__()))
            else:
                variable_dimensions[variable] = shape
    return variable_dimensions


def update_f_to_v(factor_graph: graphs.FactorGraph):
    def f_to_v_msg(f: nodes.FNode, x: nodes.VNode) -> rv.Discrete:
        msg = f.factor
        for y in factor_graph.neighbors(f):
            if y == x:
                continue
            new_message = get_message(factor_graph, y, f)
            if new_message is not None:
                msg *= new_message
        dims_to_reduce = [v for v in factor_graph.get_vnodes() if v != x]
        return msg.marginalize(*dims_to_reduce, normalize=False)

    for factor in factor_graph.get_fnodes():
        for variable in factor_graph.neighbors(factor):
            msg = f_to_v_msg(factor, variable)
            set_message(factor_graph, factor, variable, msg)


def update_v_to_f(factor_graph: graphs.FactorGraph):
    def v_to_f_msg(x: nodes.VNode, f: nodes.FNode) -> rv.Discrete:
        msg = x.init
        for h in factor_graph.neighbors(x):
            if h != f:
                new_message = get_message(factor_graph, h, x)
        return msg

    for variable in factor_graph.get_vnodes():
        for factor in factor_graph.neighbors(variable):
            msg = v_to_f_msg(variable, factor)
            set_message(factor_graph, variable, factor, msg)


def set_message(factor_graph, u, v, msg: rv.Discrete):
    (factor_graph[u][v]['object']
     .set_message(u, v, msg))


def get_message(factor_graph, u, v) -> rv.Discrete:
    return (factor_graph[u][v]['object']
            .get_message(u, v))


def sum_product(factor_graph, iterations=10000):
    # for vnode in factor_graph.get_vnodes():
    #     for fnode in factor_graph.neighbors(vnode):
    #         set_message(factor_graph, vnode, fnode, )
    for _ in range(iterations):
        update_f_to_v(factor_graph)
        update_v_to_f(factor_graph)


def get_beliefs(factor_graph: graphs.FactorGraph):
    sum_product(factor_graph)
    return [vnode.belief() for vnode in factor_graph.get_vnodes()]


def main():
    fg = graphs.FactorGraph()

    # Create variable nodes
    x1 = nodes.VNode("x1", rv.Discrete)  # with 2 states (Bernoulli)
    x2 = nodes.VNode("x2", rv.Discrete)  # with 3 states
    x3 = nodes.VNode("x3", rv.Discrete)
    x4 = nodes.VNode("x4", rv.Discrete)

    # Create factor nodes (with joint distributions)
    dist_fa = [[0.3, 0.2, 0.1],
               [0.3, 0.0, 0.1]]
    fa = nodes.FNode("fa", rv.Discrete(dist_fa, x1, x2))

    dist_fb = [[0.3, 0.2],
               [0.3, 0.0],
               [0.1, 0.1]]
    fb = nodes.FNode("fb", rv.Discrete(dist_fb, x2, x3))

    dist_fc = [[0.3, 0.2],
               [0.3, 0.0],
               [0.1, 0.1]]
    fc = nodes.FNode("fc", rv.Discrete(dist_fc, x2, x4))

    # Add nodes to factor graph
    fg.set_nodes([x1, x2, x3, x4])
    fg.set_nodes([fa, fb, fc])

    # Add edges to factor graph
    fg.set_edge(x1, fa)
    fg.set_edge(fa, x2)
    fg.set_edge(x2, fb)
    fg.set_edge(fb, x3)
    fg.set_edge(x2, fc)
    fg.set_edge(fc, x4)

    sum_product(fg)

main()