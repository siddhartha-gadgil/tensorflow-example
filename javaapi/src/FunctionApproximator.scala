package javaapi

import org.tensorflow._
import org.tensorflow.op._
import org.tensorflow.types._
import scala.util.Using
import org.tensorflow.ndarray
import org.tensorflow.framework.optimizers.Adam
import scala.util._
import org.tensorflow.op.core._
import org.tensorflow.op.math.SquaredDifference

object FunctionApproximator {
  val rnd = new Random()

  def run() = {
    Using(new Graph()) { graph =>
      def diag = { val x = rnd.nextFloat(); (x, x) }
      val funcAp = new FunctionApproximator(graph, 100)
      println("Created graph")
      val fitted =  funcAp.fit(diag, 10000)
      println("fitted")
      fitted
    }
  }.flatten
}

class FunctionApproximator(graph: Graph, dim: Int) {
  import FunctionApproximator._
  val tf: Ops = Ops.create(graph)

  val x: PlaceholderWithDefault[TFloat32] =
    tf.placeholderWithDefault(tf.constant(0f), ndarray.Shape.of())
  val y: PlaceholderWithDefault[TFloat32] =
    tf.placeholderWithDefault(tf.constant(0f), ndarray.Shape.of())
  val b1: Variable[TFloat32] = tf.variable(
    tf.constant(Array.fill(dim)(rnd.nextFloat()))
  )

  val b2: Variable[TFloat32] = tf.variable(
    tf.constant(Array.fill(dim)(rnd.nextFloat()))
  )

  val b: Variable[TFloat32] = tf.variable(
    tf.constant(rnd.nextFloat())
  )

  val A1 = tf.variable(
    tf.constant(Array.fill(dim, 1)(rnd.nextFloat()))
  )

  val A2 = tf.variable(
    tf.constant(Array.fill(dim, dim)(rnd.nextFloat()))
  )

  val A3 = tf.variable(
    tf.constant(Array.fill(1, dim)(rnd.nextFloat()))
  )

  val h1 = tf.math.tanh(
    tf.math
      .add(tf.linalg.matMul(A1, tf.reshape(x, tf.constant(Array(1, 1)))), b1)
  )

  val h2 = tf.math.tanh(tf.math.add(tf.linalg.matMul(A2, h1), b2))

  val output = tf.math.add(tf.linalg.matMul(A3, h2), b)

  val error: SquaredDifference[TFloat32] =
    tf.math.squaredDifference(output, tf.reshape(y, tf.constant(Array(1, 1))))

  val loss: ReduceSum[TFloat32] = tf.reduceSum(error, tf.constant(Array(0, 1)))

  val optimizer = new Adam(graph)

  val minimize = optimizer.minimize(loss)

  def fit(pointGen: => (Float, Float), steps: Int) = {
    Using(new Session(graph)) { session =>
      session.run(tf.init())
      (0 until steps).foreach { j =>
        val (xData, yData) = pointGen
        session
          .runner()
          .feed(x, TFloat32.scalarOf(xData))
          .feed(y, TFloat32.scalarOf(yData))
          .addTarget(minimize)
          .run()
      }
    }
  }
}
