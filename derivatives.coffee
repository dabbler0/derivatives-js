BigNumber = require 'bignumber.js'
{Vector, Matrix} = require('linear-algebra')()

Matrix.diagonal = (vector) ->
  matrix = Matrix.zero vector.cols, vector.cols
  for el, i in vector.data[0]
    matrix.data[i][i] = el
  return matrix

# EXPRESSIONS
# ===========

class Expression
  constructor: (@type) ->
    @id = Expression.nextId()
    @children ?= []

  findMatrix: (matrix) ->
    @children.map((child) -> child.findMatrix(matrix)).reduce(((a, b) -> a.concat(b)), [])

  contains: (expression) ->
    expression is @ or @children.some((child) -> child.contains(expression))

  # A matrixDerivative is a special, restricted kind
  # of derivative we can take from a scalar to a matrix.
  matrixDerivative: (point, matrix) ->
    unless @type is Type.SCALAR
      throw new ArgumentError "Cannot take a derivative of a #{@type.toString()} to a matrix"

    # Search for all the dot products
    instances = @children.map((child) -> child.findMatrix matrix).reduce((a, b) -> a.concat(b))

    result = Matrix.zero matrix.type.rows, matrix.type.cols

    for instance in instances
      result.plus_ point.value(instance.vector).trans().dot point.derivative(@, instance)

    return result

Expression.id = new BigNumber(0)
Expression.nextId = ->
  Expression.id = Expression.id.plus(1)
  return Expression.id.toString(16)

# Constant expressions

exports.VectorExpression = class VectorExpression extends Expression
  constructor: (@type) ->
    unless @type instanceof Type
      @type = new VectorType @type

    super @type

  contains: (x) -> x is @

  findMatrix: -> []

  value: (point) -> point.value @

  derivative: (point, x) ->
    if x is @
      return Matrix.identity @type.length
    else
      return Matrix.zero @type.length, @type.length

exports.MatrixExpression = class MatrixExpression extends Expression
  constructor: (@type, cols) ->
    unless @type instanceof Type
      @type = new MatrixType @type, cols

    super @type

  contains: (x) -> x is @

  value: (point) -> point.value @

  derivative: (point, x) ->

class TransformExpression extends Expression
  constructor: (@vector, @matrix) ->
    @children = [@vector, @matrix]
    @type = new VectorType @matrix.type.cols
    super @type

  findMatrix: (matrix) ->
    result = super matrix
    if matrix is @matrix
      result.push @
    return result

  value: (point) -> point.value(@vector).dot(point.value(@matrix))

  derivative: (point, x) ->
    if @vector.contains x
      return point.value(@matrix).trans().dot point.derivative @vector, x
    else
      return Matrix.zero @type.length, x.type.length

class DotExpression extends Expression
  constructor: (@left, @right) ->
    @children = [@left, @right]
    @type = Type.SCALAR
    super @type

  value: (point) -> point.value(@left).dot(point.value(@right).trans())

  derivative: (point, x) ->
    result = Matrix.zero 1, x.type.length

    if @left.contains x
      result.plus_ point.value(@right).dot point.derivative(@left, x)
    if @right.contains x
      result.plus_ point.value(@left).dot point.derivative(@right, x)

    return result

class AddExpression extends Expression
  constructor: (@left, @right) ->
    @children = [@left, @right]
    @type = @left.type
    super @type

  value: (point) -> point.value(@left).plus point.value(@right)

  derivative: (point, x) ->
    result = Matrix.zero @type.length, x.type.length

    if @left.contains x
      result.plus_ point.derivative @left, x
    if @right.contains x
      result.plus_ point.derivative @right, x

    return result

class MulExpression extends Expression
  constructor: (@left, @right) ->
    @children = [@left, @right]
    @type = @left.type
    super @type

  value: (point) -> point.value(@left).mul point.value(@right)

  derivative: (point, x) ->
    result = Matrix.zero @type.length, x.type.length

    if @left.contains x
      result.plus_ Matrix.diagonal(point.value(@right)).dot point.derivative @left, x
    if @right.contains x
      result.plus_ Matrix.diagonal(point.value(@left)).dot point.derivative @right, x

    return result

class MapExpression extends Expression
  constructor: (@base, @function) ->
    unless @function instanceof Differentiable
      @function = new Differentiable @function

    @children = [@base]
    @type = @base.type
    super @type

  value: (point) -> point.value(@base).map (x) => @function.value x

  derivative: (point, x) ->
    if @base.contains x
      return Matrix.diagonal(point.value(@base).map((x) => @function.derivative(x))).dot point.derivative @base, x
    else
      return Matrix.zero @type.length, x.type.length

# POINT
# =====
exports.Point = class Point
  constructor: (pairs) ->
    @cache = {}

    for pair in pairs
      @cache[pair[0].id] = pair[1]

    @diffCache = {}

  value: (expression) ->
    # Evaluate this point if it has not yet been evaluated
    unless expression.id of @cache
      @cache[expression.id] = expression.value @

    return @cache[expression.id]

  derivative: (z, wrt) ->
    if z is wrt
      return Matrix.identity(z.type.length)

    # Evaluate this point if it has not yet been evaluated
    unless z.id of @diffCache
      @diffCache[z.id] = {}
    unless wrt.id of @diffCache[z.id]
      @diffCache[z.id][wrt.id] = z.derivative @, wrt

    return @diffCache[z.id][wrt.id]

# TYPES
# =====
class Type
  constructor: (@form) ->

class VectorType extends Type
  constructor: (@length) ->
    @form = 'vector'

  equal: (other) ->
    other.form is @form and other.length is @length

class MatrixType extends Type
  constructor: (@rows, @cols) ->
    @form = 'matrix'

  equal: (other) ->
    other.form is @form and other.rows is @rows and other.cols is @cols

Type.SCALAR = new VectorType 1

# DIFFERENTIABLE
# ==============
class Differentiable
  constructor: (@value, @derivative) ->
    unless @derivative?
      @derivative = (x) => @value(x + _epsilon) - @value(x - _epsilon) / (2 * _epsilon)

_epsilon = 10 ** -5

exports.fun = fun = {
  constant: (c) -> new Differentiable((-> c), (-> 0))
  times: (c) -> new Differentiable(((x) -> x * c), (-> c))
  sigmoid: new Differentiable(
    ((x) -> 1 / (1 + Math.E ** -x)),
    ((x) ->
      s = 1 / (1 + Math.E ** -x)
      return s * (1 - s)
    )
  )
}
exports.ops = {
  sum: (x) -> new DotExpression x, new MapExpression x, fun.constant(1)
  dot: (a, b) -> new DotExpression a, b
  add: (a, b) -> new AddExpression a, b
  mul: (a, b) -> new MulExpression a, b
  map: (a, b) -> new MapExpression a, b
  scale: (base, constant) -> new MapExpression base, fun.times(constant)
  transform: (a, b) -> new TransformExpression a, b
}
