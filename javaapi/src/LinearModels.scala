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

  val simpleData = (0 to 15000).map { n =>
    val x = rnd.nextGaussian().toFloat * 10
    val y = 2 * x + 1
    (x, y)
  }

  val noisyData = (0 to 15000).map { n =>
    val x = rnd.nextGaussian().toFloat * 10
    val y = 2 * x + 1 + rnd.nextGaussian().toFloat
    (x, y)
  }

  def run(): Unit = {
    Using(new Graph()) { graph =>
      println("Running simple model")
      val simpleModel = new SimpleLinearModel(graph, 0.1f)
      simpleModel.fit(noisyData)
    }
    Using(new Graph()) { graph =>
      val batchModel = new BatchLinearModel(graph, 0.1f)
      println("Fitting batch model")
      batchModel.fit(noisyData, 20)
    }
  }
}

// fit for y = mx + c feeding in values of x and y as real numbers
class SimpleLinearModel(graph: Graph, learningRate: Float) {
  val tf = Ops.create(graph)

  val optimizer = new AdaGrad(graph, learningRate)

  val m = tf.variable(tf.constant(0.1f))
  val c = tf.variable(tf.constant(0f))

  val x = tf.withName("X").placeholder(TFloat32.DTYPE)
  val y = tf.withName("Y").placeholder(TFloat32.DTYPE)

  val shape0 = Shape.of(1)

  val loss = tf.math.squaredDifference(y, tf.math.add(tf.math.mul(m, x), c))

  val minimize = optimizer.minimize(loss)

  def fit(xy: Seq[(Float, Float)]) = Using(new Session(graph)) { session =>
    session.run(tf.init())
    xy.foreach { case (xdata, ydata) =>
      val xTensor = TFloat32.tensorOf(StdArrays.ndCopyOf(Array(xdata)))
      val yTensor = TFloat32.tensorOf(StdArrays.ndCopyOf(Array(ydata)))
      session
        .runner()
        .feed(x, xTensor)
        .feed(y, yTensor)
        .addTarget(minimize)
        .run()
    }
    println(s"Got m = ${opLookup(m, session)} and c = ${opLookup(c, session)}")
  }
}

class BatchLinearModel(graph: Graph, learningRate: Float) {
  val tf = Ops.create(graph)

  val optimizer = new AdaGrad(graph, learningRate)

  val m = tf.variable(tf.constant(Array(Array(0.1f))))
  val c = tf.variable(tf.constant(Array(Array(0.0f))))

  val x = tf.withName("X").placeholder(TFloat32.DTYPE)
  val y = tf.withName("Y").placeholder(TFloat32.DTYPE)

  val shape0 = Shape.of(1)

  val const = tf.onesLike(x)

  val prediction =
    tf.math.add(tf.linalg.matMul(m, x), tf.linalg.matMul(c, const))

  val loss =
    tf.reduceSum(tf.math.squaredDifference(y, prediction), tf.constant(1))

  val minimize = optimizer.minimize(loss)

  def fit(xy: Seq[(Float, Float)], batchSize: Int) = Using(new Session(graph)) {
    session =>
      session.run(tf.init())
      val groups = xy.toVector.grouped(batchSize).toVector
      groups.foreach { v =>
        val xData = v.map(_._1).toArray
        val yData = v.map(_._2).toArray
        val xTensor = TFloat32.tensorOf(StdArrays.ndCopyOf(Array(xData)))
        val yTensor = TFloat32.tensorOf(StdArrays.ndCopyOf(Array(yData)))
        // val output =
        session
          .runner()
          .feed(x, xTensor)
          .feed(y, yTensor)
          .addTarget(minimize)
          // .fetch(const)
          .run()
      // val data = output.get(0).expect(TFloat32.DTYPE).data()
      // val size = data.size().toInt
      // scala.util.Try(println((0 until size).toVector.map(j => data.getFloat(0, j))))
      // println(size)

      }
      println(
        s"Got m = ${opLookup(m, session)} and c = ${opLookup(c, session)}"
      )
  }
}
