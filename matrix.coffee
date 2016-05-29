{Vector, Matrix} = require 'linear-algebra'

class MatrixType
  constructor: (@rows, @cols) ->

# Differentiables
# ===============

# MatrixDifferentiable
# --------------------
# A differentiable scalar function of a matrix
# Ideally, we would be provided with exact derivatives;
# if we do not have them, we can approximate them.
class MatrixDifferentiable
  constructor: (@value, @grad) ->

MatrixDifferentiable.approximate = (fn) ->
  grad = (matrix) ->
    base = fn matrix
    results = Matrix.zeroes(matrix.rows, matrix.cols)

    for row, i in matrix.data
      for cell, j in row
        matrix.data[i][j] += MatrixDifferentiable._epsilon
        results.data[i][j] = (fn(matrix) - base) / _epsilon

    return results

  return new MatrixDifferentiable fn, grad

MatrixDifferentiable._epsilon = 10 ** -5

# ScalarDifferentiable
# --------------------
# Same thing, but for a scalar function of a scalar
class ScalarDifferentiable
  constructor: (@value, @grad) ->

ScalarDifferentiable.approximate = (fn) ->
  grad = (x) ->
    return (fn(x + ScalarDifferentiable._epsilon) - fn(x)) / ScalarDifferentiable._epsilon

  return new ScalarDifferentiable fn, grad

ScalarDifferentiable._epsilon = 10 ** -5

# EXPRESSIONS
# ===========
class Expression
  constructor: ->
    @id = Expression.nextId()

Expression.id = new BigNumber(0)
Expression.nextId = ->
  Expression.id = Expression.id.plus(1)
  return Expression.id.toString(16)

# ScalarExpression
# ----------------
# A ScalarExpression is a scalar extracted somehow from a matrix.
# @function should be of type MatrixDifferentiable or Function
class ScalarExpression extends Expression
  constructor: (@base, @function) ->
    # Convert @function to a Differentiable if it is not already
    if @function instanceof Function
      @function = MatrixDifferentiable.approximate @function

    @type = new MatrixType 1, 1

    super

  value: (point) -> @function.value point.get @base

  derivative: (point, x) ->
    if x is @
      return matrix.fill(@type, 1)

    result = Matrix.zeroes(x.type.rows, x.type.cols)

    # Compute top-level d(value)/d(@base)
    baseGrad = @function.grad point.get @base

    # Compute d(@base)/d(x) d(value)/d(@base)
    return @base.derivative(point, x).chain(baseGrad)

# DotExpression
# -------------
# A DotExpression takes two matrices and multiplies them.
class DotExpression extends Expression
  constructor: (@left, @right) ->
    @type = new MatrixType @left.type.rows, @right.type.cols

    super

  value: (point) -> point.get(@left).dot(point.get(@right))

  derivative: (point, x) ->
    # Compute the derivatives here wrt left and right
    dLeft = new Derivative(
      point.get(@left).trans()
      Matrix.identity(@right.cols)
    )
    dRight = new Derivative(
      Matrix.identity(@left.rows)
      point.get(@right).trans()
    )

    # Compute the net derivative as the sum of the derivatives
    # dZ/dA * dA/dX + dZ/dB * dB/dX
    return dLeft.dot(@left.derivative(point, x)).add(
      dRight.dot(@right.derivative(point, x))
    )

# MulExpression
# -------------
# A MulExpression applies a Hadamard product
class MulExpression extends Expression
  constructor: (@left, @right) ->
    @type = @left.type

    super

  value: (point) ->
    point.get(@left).mul @right

  derivative: (point, x) ->

# A Derivative is a second-order summary of a fourth-order tensor representing the derivative of a matrix with respect to another.
# Derivative is such that @side corresponds to the side the matrix was on in a matrix multiplication.
# For instance:
#   z (scalar cost) = f(Z)
#   Z = ABCD = (zxa)(axb)(bxc)(cxd) => (zxd)
#   We want to compute dZ/dC using the following association
#   Z = [A[B[C[D]]]]
#     dZ/d(BCD) = A = <T(A), I(d)>
#     d(BCD)/d(CD) = B = <T(B), I(d)>
#     d/(CD)/dC = D = <I(b), T(D)>
#
#   We then combine these top to bottom, adding on the left or right as indicated
#   (zxd)
#   (axz) (zxd) (LEFT) => (axd), the dimensions of (BCD)
#   (bxa) (axz) (zxd) (LEFT) => (bxd), the dimensions of (CD)
#   (bxa) (axz) (zxd) (dxc) (RIGHT) => (bxc), the dimensions of (C)
#
# What about maps and Hadamard products? (They're basically the same)
# Suppose we inserted a Hadamard product by G in front of B.
# Then we would get:
#   Z = A(G)[BCD]
# And then what would we want our staircase to look like? Note that G has dimensions (axd).
#   (zxd)
#   (axz) (zxd) (dxd)
#   (axz) (zxd) (dxd) HADAMARD (axd)
class Derivative
  constructor: (@left, @right) ->

  dot: (other) ->
    return new Derivative(
      other.left.dot(@left),
      @right.dot(other.right)
    )

  plus: (other) ->
    return new Derivative(
      other.left.plus(@left),
      @right.plus(other.right)
    )

  scalarize: (scalar) ->
    return @left.dot(scalar).dot(@right)
