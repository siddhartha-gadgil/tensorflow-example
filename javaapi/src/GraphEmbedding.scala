package javaapi

import org.tensorflow._, ndarray._, types._
import org.tensorflow.op._, core.Variable
import org.tensorflow.op.core.Placeholder
import org.tensorflow.op.math.Add
import org.tensorflow.types._, family.TType
import scala.util.Using
import org.tensorflow.ndarray._
import org.tensorflow.framework.optimizers.{
  Optimizer,
  GradientDescent,
  AdaGrad,
  AdaDelta,
  Adam
}
import scala.jdk.CollectionConverters._
import org.tensorflow.framework.optimizers.AdaGrad

import Optimizer.GradAndVar
import scala.jdk.CollectionConverters._
import Utils._
import scala.util._
import GraphEmbedding._

object GraphEmbedding {
  val rnd = new Random()
  def run() = Using(new Graph()) { graph =>
    println("running graph embedding")
    val g = new GraphEmbedding(6, graph)
    g.test()
  }
}

class GraphEmbedding(numPoints: Int, graph: Graph, epsilon: Float = 0.01f) {
  val tf = Ops.create(graph)

  val ones = tf.constant(Array.fill(numPoints)(1.0f))

  val xs = tf.variable(
    tf.constant(Array.fill(numPoints)(rnd.nextFloat() * 2.0f))
  )

  val ys = tf.variable(
    tf.constant(Array.fill(numPoints)(rnd.nextFloat() * 2.0f))
  )

  def rankOne(v: Operand[TFloat32], w: Operand[TFloat32]) =
    tf.linalg.matMul(
      tf.reshape(v, tf.constant(Array(numPoints, 1))),
      tf.reshape(w, tf.constant(Array(1, numPoints)))
    )

  val xDiff = tf.math.squaredDifference(rankOne(xs, ones), rankOne(ones, xs))

  val yDiff = tf.math.squaredDifference(rankOne(ys, ones), rankOne(ones, ys))

  val totDiff = tf.math.add(xDiff, yDiff)

  val probs = tf.math.div(
    tf.constant(1.0f),
    tf.math.add(
      tf.constant(1.0f + epsilon),
      totDiff
    )
  )

  val incidence = tf.placeholder(TFloat32.DTYPE)

  val loss = tf.math.neg(
    tf.math.add(
      tf.math.mul(incidence, tf.math.log(probs)),
      tf.math.mul(
        tf.math.sub(tf.constant(1.0f), incidence),
        tf.math.sub(tf.constant(1.0f), tf.math.log(probs))
      )
    )
  )

  def test() = {
    Using(new Session(graph)) { session =>
      session.run(tf.init())
      println("initialized")
      val pd = dataLookup(probs, session)
      val pmat = Vector.tabulate(numPoints, numPoints) { case (i, j) =>
        pd.getFloat(i, j)
      }
      println(pmat.mkString("\n"))
    }
  }.fold(fa => println(fa.getMessage), (_) => println("Done"))
}
