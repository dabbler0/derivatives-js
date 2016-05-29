BigNumber = require 'bignumber.js'
numeric = require 'numeric'

_epsilon = 10 ** -5

class Expression
  constructor: (params, @f) ->
    @params = params.map (x, i) =>
      {
        expression: x
        derivative: (point) =>
          # Approximate the derivative, by default
          p = params.map((param) -> point.get(param))

          p[i] += _epsilon; over = @f.apply(@, p)
          p[i] -= 2 * _epsilon; under = @f.apply(@, p)

          return (over - under) / (2 * _epsilon)
      }

    @id = Expression.nextId()

  derivative: (point, x) ->
    if x is @
      return 1

    # Multivariate chain rule
    result = 0
    for y in @params
      result += y.derivative(point) * y.expression.derivative(point, x)
    return Math.round result # DEBUGGING

Expression.id = new BigNumber(0)
Expression.nextId = ->
  Expression.id = Expression.id.plus(1)
  return Expression.id.toString(16)

class InputExpression extends Expression
  constructor: ->
    @id = Expression.nextId()

  derivative: (point, x) ->
    if @ is x
      return 1
    else
      return 0

class Point
  constructor: (pairs) ->
    @cache = {}

    for pair in pairs
      console.log pair[0].id, pair[1]
      @cache[pair[0].id] = pair[1]

  get: (expression) ->
    # Evaluate this point if it has not yet been evaluated
    unless expression.id of @cache
      @cache[expression.id] = expression.f.apply(expression,
        expression.params.map((param) => @get(param.expression)))

    return @cache[expression.id]

###
x = new InputExpression()
y = new InputExpression()

z = new Expression([x, y], (x, y) -> x * y)
n = new Expression([x, y], (x, y) -> x + y)

u = new Expression([z], (z) -> z * z)
w = new Expression([n], (n) -> n * n)

complex = new Expression([u, w, n, z], (u, w, n, z) -> u / w + n / z)

point = new Point([
  [x, 3]
  [y, 5]
])

console.log 'du/dz=', u.derivative(point, z), 'dz/dy=', z.derivative(point, y), '=>', u.derivative(point, y)
console.log 'dw/dn=', w.derivative(point, n), 'dn/dy=', n.derivative(point, y), '=>', w.derivative(point, y)
# This should theoretically be
# u = x^2 y^2
# du/dy = 2 y x^2 = 2 * 9 * 5 = 90
#
# w = x^2 + 2xy + y^2
# dw/dy = 2y + 2x = 16

# And now we try some gradient descent:
currentX = 3
currentY = 5
for [0..100]
  point = new Point [
    [x, currentX]
    [y, currentY]
  ]
  console.log 'X', currentX, 'Y', currentY, 'Current cost:', point.get(complex)
  dx = complex.derivative point, x
  dy = complex.derivative point, y
  #console.log currentX, currentY, 'dC/dx=', dx = complex.derivative(point, x), 'dC/dy=', dy = complex.derivative(point, y)
  currentX += dx * -0.1
  currentY += dy * -0.1

# This should find the local minimum 1.88988 (according to Wolfram Alpha)
###

matrix = (((new InputExpression()) for [1..3]) for [1..4])
vector = ((new InputExpression() for [1..2]) for [1..3])

params = matrix.reduce((a, b) -> Array.prototype.concat(a, b)).concat vector.reduce((a, b) -> Array.prototype.concat(a, b))

# Unpack
[
  [a, b, c]
  [d, e, f]
  [g, h, i]
  [j, k, l]
] = matrix

[
  [m, n]
  [o, p]
  [q, r]
] = vector

dotFunction = (_i, _j) ->
  return (
      a, b, c,
      d, e, f,
      g, h, i,
      j, k, l,

      m, n,
      o, p,
      q, r) ->
    reassembledMatrix = [
      [a, b, c]
      [d, e, f]
      [g, h, i]
      [j, k, l]
    ]
    reassembledVector = [
      [m, n]
      [o, p]
      [q, r]
    ]

    return numeric.dot(reassembledMatrix, reassembledVector)[_i][_j]

product = ((new Expression(params, dotFunction(_i, _j)) for _j in [0...2]) for _i in [0...4])

point = new Point [
  # Matrix
  [a, 1], [b, 2], [c, 3]
  [d, 4], [e, 5], [f, 6]
  [g, 7], [h, 8], [i, 9]
  [j, 10], [k, 11], [l, 12]

  # Vector
  [m, 13], [n, 14]
  [o, 15], [p, 16]
  [q, 17], [r, 18]
]

console.log 'MATRIX'
console.log matrix.map (row) -> row.map (el) -> point.get el
console.log 'VECTOR'
console.log vector.map (row) -> row.map (el) -> point.get el
console.log 'PRODUCT'
console.log product.map (row) -> row.map (el) -> point.get(el)

console.log 'FOURTH-DIMENSIONAL TENSOR DERIVATIVE'
console.log product.map((prow) -> prow.map((pcell) ->
  JSON.stringify(matrix.map (row) -> row.map (cell) -> pcell.derivative(point, cell))).join('\t')).join('\n')
console.log 'OTHER FOURTH-DIMENSIONAL TENSOR DERIVATIVE'
console.log product.map((prow) -> prow.map((pcell) ->
  JSON.stringify(vector.map (row) -> row.map (cell) -> pcell.derivative(point, cell))).join('\t')).join('\n')
console.log 'SUMMARIZED MATRIX DERIVATIVE'
console.log summary = numeric.transpose vector.map (row) -> row.map (el) -> point.get el
console.log 'EXAMPLE MULTIPLICATION WITH SUMMARIZED MATRIX DERIVATIVE'
console.log numeric.dot [
  [1, 0]
  [0, 0]
  [0, 0]
  [0, 0]
], summary
console.log numeric.dot [
  [1, .1]
  [2, .3]
  [.01, 4]
  [2, 0]
], summary
console.log 'OTHER SUMMARIZED MATRIX DERIVATIVE'
console.log summary = numeric.transpose matrix.map (row) -> row.map (el) -> point.get el
console.log 'EXAMPLE MULTIPLICATION WITH OTHER SUMMARIZED MATRIX DERIVATIVE'
console.log numeric.dot summary, [
  [1, 0]
  [0, 0]
  [0, 0]
  [0, 0]
]
console.log numeric.dot summary, [
  [1, 1]
  [1, 1]
  [1, 1]
  [1, 1]
]
