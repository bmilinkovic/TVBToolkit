"""Explicit experimental derivative attenuation; NOT a unit conversion.

Transform only the nested finite-difference denominators in the existing native
second-order RHS, retaining its covariance indexing and all other equations.
Fail closed if the expected helper layout changes. No production class is edited.
"""
import ast
import inspect
import textwrap
from types import MethodType


def scaled_rhs(model, scale=1000.):
    original=type(model).dfun
    # Some native comments start at column zero, defeating textwrap.dedent.
    source=inspect.getsource(original)
    method=ast.parse('class _Source:\n'+source).body[0].body[0]
    tree=ast.Module(body=[method],type_ignores=[])
    expected={'_diff_fe':scale,'_diff_fi':scale,
              '_diff2_fe_fe_e':scale**2,'_diff2_fe_fe_i':scale**2,
              '_diff2_fi_fi_e':scale**2,'_diff2_fi_fi_i':scale**2,
              # Inner first derivative is already /scale; scale outer again.
              '_diff2_fi_fe':scale,'_diff2_fe_fi':scale}
    found=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.FunctionDef) and node.name in expected:
            returns=[n for n in node.body if isinstance(n,ast.Return)]
            if len(returns)!=1 or not isinstance(returns[0].value,ast.BinOp) or not isinstance(returns[0].value.op,ast.Div):
                raise RuntimeError('Unexpected finite-difference helper; audit adapter')
            expr=returns[0].value
            expr.right=ast.BinOp(left=expr.right,op=ast.Mult(),right=ast.Constant(expected[node.name]))
            found.append(node.name)
    if set(found)!=set(expected): raise RuntimeError('Missing derivative helpers')
    ast.fix_missing_locations(tree)
    scope=dict(original.__globals__)
    exec(compile(tree,'<experimental_scaled_second_order>','exec'),scope)
    model.dfun=MethodType(scope['dfun'],model)
    return found
