package javaapi

import org.tensorflow._, ndarray._
import org.tensorflow.op._, core.Variable
import org.tensorflow.op.core.Placeholder
import org.tensorflow.op.math.Add
import org.tensorflow.types._
import scala.util.Using
import org.tensorflow.ndarray._
import org.tensorflow.framework.optimizers.{Optimizer, GradientDescent, AdaGrad}
import scala.jdk.CollectionConverters._
import GeometricSimple.opLookup
import org.tensorflow.framework.optimizers.AdaGrad

import GeometricSimple.opLookup

object SimpleLinearModel {
  val rnd = new scala.util.Random()

  val simpleData = (0 to 1500).map { n =>
    val x = rnd.nextGaussian().toFloat * 10
    val y = 2 * x + 1
    (x, y)
  }

  def run(): Unit = Using(new Graph()) { graph =>
    println("Running simple model")
    val simpleModel = SimpleLinearModel(graph, 0.1f)
    simpleModel.fit(simpleData)
  }
}

// fit for y = mx + c feeding in values of x and y as real numbers
case class SimpleLinearModel(graph: Graph, learningRate: Float) {
  val tf = Ops.create(graph)

  val optimizer = new AdaGrad(graph, learningRate)

  val m = tf.variable(tf.constant(1f))
  val c = tf.variable(tf.constant(0f))

  val x = tf.withName("X").placeholder(TFloat32.DTYPE)
  val y = tf.withName("Y").placeholder(TFloat32.DTYPE)

  val shape0 = Shape.of(1)

  val loss = tf.math.squaredDifference(y, tf.math.add(tf.math.mul(m, x), c))

  val minimize = optimizer.minimize(loss)

  def fit(xy: Seq[(Float, Float)]) = Using(new Session(graph)) { session =>
    session.run(tf.init())
    xy.foreach { case (xdata, ydata) =>
      println(s"feeding $xdata, $ydata")
      val xTensor = TFloat32.tensorOf(StdArrays.ndCopyOf(Array(xdata)))
      val yTensor = TFloat32.tensorOf(StdArrays.ndCopyOf(Array(ydata)))
      session
        .runner()
        .feed(x, xTensor)
        .feed(y, yTensor)
        .addTarget(minimize)
        .run()
      println(
        s"After feeding y: $ydata and x: $xdata, got m = ${opLookup(m, session)} and c = ${opLookup(c, session)}"
      )
    }
  }
}
