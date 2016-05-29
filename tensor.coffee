BigNumber = require 'bignumber.js'
{Matrix, Vector} = require('linear-algebra')()

# The most general, and the slowest, of them all.
# Henceforth everything is cheats. This is what we test against.
class Expression
  constructor: (@params, @function, @type) ->
    @id = Expression.nextId()

  value: (point) ->
    paramValues = @params.map (param) -> point.get param
    return @function.apply @, paramValues

  derivative: (point, x) ->
    if x is @
      result = Tensor4 new Type [@type.size[0], @type.size[1], @type.size[0], @type.size[1]]
      for i in [0...@type.size[0]]
        for j in [0...@type.size[1]]
          result.data[i][j].data[i][j] = 1
      return result

    result = new Tensor4 new Type [x.type.size[0], x.type.size[1], @type.size[0], @type.size[1]]

    paramValues = @params.map (param) -> point.get param

    base = @function.apply @, paramValues

    # Compute the derivative with respect to this parameter
    @params.forEach (param, paramIndex) =>
      derivative = new Tensor4 new Type [param.type.size[0], param.type.size[1], @type.size[0], @type.size[1]]

      # Estimate each of the derivatives
      for i in [0...paramValues[paramIndex].rows]
        for j in [0...paramValues[paramIndex].cols]
          paramValues[paramIndex].data[i][j] += Expression._epsilon

          partial = @function.apply(@, paramValues).minus(base).mulEach 1 / Expression._epsilon

          paramValues[paramIndex].data[i][j] -= Expression._epsilon

          for k in [0...partial.rows]
            for l in [0...partial.cols]
              derivative.data[k][l].data[i][j] = partial.data[k][l]

      # Compute the derivative of the parameter and add it to the final result
      result.plus_ param.derivative(point, x).dot(derivative)

    return result

class InputExpression extends Expression
  constructor: (@type) ->
    @id = Expression.nextId()

  derivative: (point, x) ->
    result = new Tensor4 new Type [x.type.size[0], x.type.size[1], @type.size[0], @type.size[1]]

    if x is @
      for i in [0...@type.size[0]]
        for j in [0...@type.size[1]]
          result.data[i][j].data[i][j] = 1

    return result

Expression._epsilon = 10 ** -5
Expression.id = new BigNumber(0)
Expression.nextId = ->
  Expression.id = Expression.id.plus(1)
  return Expression.id.toString(16)

class Type
  constructor: (@size) ->

class Tensor4
  constructor: (@type) ->
    @data = (for i in [0...@type.size[2]]
      for j in [0...@type.size[3]]
        Matrix.zero(@type.size[0], @type.size[1]))

  _dotMatrix: (matrix) ->
    unless matrix.rows is @type.size[2] and matrix.cols is @type.size[3]
      throw new ArgumentError "Cannot multiply Matrix <#{matrix.rows}, #{matrix.cols}> with Tensor4 #{@type.toString()}"

    result = Matrix.zero @type.size[0], @type.size[1]
    for i in [0...matrix.rows]
      for j in [0...matrix.cols]
        result.plus_ @data[i][j].mulEach matrix.data[i][j]

    return result

  _dotTensor: (other) ->
    # Test if they are multiplicable
    unless other.type.size[0] is @type.size[2] and other.type.size[1] is @type.size[3]
      throw new ArgumentError "Cannot chain-multiply Tensor4 types #{other.type.toString()}, #{@type.toString()}"

    resultType = new Type [
      @type.size[0]
      @type.size[1]
      other.type.size[2]
      other.type.size[3]
    ]

    result = new Tensor4 resultType

    for i in [0...other.type.size[2]]
      for j in [0...other.type.size[3]]
        result.data[i][j].plus_ @_dotMatrix other.data[i][j]

    return result

  dot: (other) ->
    if other instanceof Tensor4
      return @_dotTensor other
    else if other instanceof Matrix
      return @_dotMatrix other

  plus_: (other) ->
    for i in [0...@type.size[2]]
      for j in [0...@type.size[3]]
        @data[i][j].plus_ other.data[i][j]

  plus: (other) ->
    return (for i in [0...@type.size[2]]
      for j in [0...@type.size[3]]
        @data[i][j].plus other.data[i][j])

  print: ->
    str = ''
    totalRows = @type.size[0] * @type.size[2]
    totalCols = @type.size[1] * @type.size[3]

    for i in [0...totalRows]
      if i % @type.size[0] is 0
        for [0...totalCols]
          str += '-\t'
        str += '\n'
      for j in [0...totalCols]
        if j % @type.size[1] is 0
          str += '|'
        str += Math.round(@data[Math.floor i / @type.size[0]][Math.floor j / @type.size[1]].data[i % @type.size[0]][j % @type.size[1]]) + '\t'
      str += '\n'

    return str

class Point
  constructor: (pairs) ->
    @cache = {}

    for pair in pairs
      @cache[pair[0].id] = pair[1]

  get: (expression) ->
    # Evaluate this point if it has not yet been evaluated
    unless expression.id of @cache
      @cache[expression.id] = expression.value @

    return @cache[expression.id]

A = new InputExpression new Type [4, 3]
B = new InputExpression new Type [3, 2]
G = new InputExpression new Type [4, 2]
C = new InputExpression new Type [2, 1]

AB = new Expression [A, B], ((a, b) -> a.dot(b)), new Type [4, 2]
ABG = new Expression [AB, G], ((ab, g) -> ab.mul(g)), new Type [4, 2]
ABGS = new Expression [ABG], ((abg) -> abg.map((x) -> x ** 2)), new Type [4, 2]
ABGC = new Expression [ABGS, C], ((abgs, c) -> abgs.dot(c)), new Type [4, 1]

ABGC_TRUE = new Expression [A, B, G, C], ((a, b, g, c) -> a.dot(b).mul(g).map((x) -> x ** 2).dot(c)), new Type [4, 1]

point = new Point [
  [A, new Matrix [
    [1, 2, 3]
    [4, 5, 6]
    [7, 8, 9]
    [10, 11, 12]
  ]],

  [B, new Matrix [
    [13, 14]
    [15, 16]
    [17, 18]
  ]],

  [G, new Matrix [
    [1, -1]
    [-1, 1]
    [1, 1]
    [-1, -1]
  ]],

  [C, new Matrix [
    [19]
    [20]
  ]]
]

console.log 'COMPARE'
console.log ABGC.derivative(point, A).print()
console.log ABGC_TRUE.derivative(point, A).print()
console.log 'COMPARE'
console.log ABGC.derivative(point, B).print()
console.log ABGC_TRUE.derivative(point, B).print()
console.log 'COMPARE'
console.log ABGC.derivative(point, G).print()
console.log ABGC_TRUE.derivative(point, G).print()
console.log 'COMPARE'
console.log ABGC.derivative(point, C).print()
console.log ABGC_TRUE.derivative(point, C).print()
