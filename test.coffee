{Matrix, Vector} = require('linear-algebra')()
{Point, VectorExpression, MatrixExpression, fun, ops} = require './derivatives.coffee'

# Input
input0 = new VectorExpression 10

# Some parameters
weight1 = new VectorExpression 10
bias1 = new VectorExpression 10
layer1 = new MatrixExpression 10, 20

reweighted1 = ops.add ops.mul(input0, weight1), bias1
rectified1 = ops.map reweighted1, fun.sigmoid
input1 = ops.transform rectified1, layer1

weight2 = new VectorExpression 20
bias2 = new VectorExpression 20
layer2 = new MatrixExpression 20, 2

reweighted2 = ops.add ops.mul(input1, weight2), bias2
rectified2 = ops.map reweighted2, fun.sigmoid

input2 = ops.transform rectified2, layer2

output = ops.map input2, fun.sigmoid

wantedOutput = new VectorExpression 2

point = new Point [
  [input0, input0Value = Vector.zero(10).map(-> Math.random())]
  [weight1, weight1Value = Vector.zero(10).map(-> Math.random())]
  [bias1, bias1Value = Vector.zero(10).map(-> Math.random())]
  [layer1, layer1Value = Matrix.zero(10, 20).map(-> Math.random())]
  [weight2, weight2Value = Vector.zero(20).map(-> Math.random())]
  [bias2, bias2Value = Vector.zero(20).map(-> Math.random())]
  [layer2,layer2Value =  Matrix.zero(20, 2).map(-> Math.random())]
  [wantedOutput, wantedOutputValue = Vector.zero(2).map -> Math.round Math.random()]
]


diff = ops.add wantedOutput, ops.scale output, -1
cost = ops.dot diff, diff

for [1..1000]
  console.log 'WANTED OUTPUT', point.value(wantedOutput).data[0]
  console.log 'ACTUAL OUTPUT', point.value(output).data[0]
  console.log 'CURRENT COST', point.value(cost).data[0]

  grad = cost.derivative point, rectified2
  console.log grad
  bias2Value.plus_ grad.mulEach -100
  console.log bias2Value
  point = new Point [
    [input0, input0Value]
    [weight1, weight1Value]
    [bias1, bias1Value]
    [layer1, layer1Value]
    [weight2, weight2Value]
    [bias2, bias2Value]
    [layer2,layer2Value]
    [wantedOutput, wantedOutputValue]
  ]
