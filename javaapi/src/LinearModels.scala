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

object SimpleLinearModel {
  val rnd = new scala.util.Random()

  val simpleData = (0 to 15000).map { n =>
    val x = rnd.nextGaussian().toFloat * 10
    val y = 2 * x + 1
    (x, y)
  }

  val mixedData = (0 to 50000).map { n =>
    val x = rnd.nextGaussian().toFloat * 10
    val parity = if (n % 2 == 0) 1 else -1
    val y = 2 * x + parity + rnd.nextGaussian().toFloat
    (x, y)
  }

  val noisyData = (0 to 20000).map { n =>
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
    Using(new Graph()) { graph =>
      val forkedModel = new ForkedLinearModel(graph, 0.1f)
      println("Fitting forked model")
      forkedModel.fit(mixedData)
      println("saved model")
    }
    println("Seeking bundle")
    val savedModelBundle =
      SavedModelBundle.load("model", SavedModelBundle.DEFAULT_TAG)
    println("Got bundle, seeking m by looking up in saved session")
    println(namedLookup("M", savedModelBundle.session()))
    val output = savedModelBundle
      .session()
      .runner()
      .feed("X", TFloat32.scalarOf(1.0f))
      .feed("Y", TFloat32.scalarOf(2.0f))
      .fetch("loss1")
      .addTarget("minimize1")
      .run()
      .get(0)
      .expect(TFloat32.DTYPE)
      .data()
      .getFloat()
    println(s"Loss: $output")
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
      val xTensor = TFloat32.scalarOf(xdata)
      val yTensor = TFloat32.scalarOf(ydata)
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

  val optimizer = new Adam(graph, learningRate)

  val m0 = tf.variable(tf.constant(0.1f))

  val m = tf.reshape(m0, tf.constant(Shape.of(1, 1)))
  // tf.variable(tf.constant(Array(Array(0.1f))))
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
        session
          .runner()
          .feed(x, xTensor)
          .feed(y, yTensor)
          .addTarget(minimize)
          .run()
      // session.save("bundle/variables")
      }
      println(
        s"Got m = ${opLookup(m, session)} and c = ${opLookup(c, session)}"
      )
  }
}

class ForkedLinearModel(graph: Graph, learningRate: Float) {
  val tf = Ops.create(graph)

  val optimizer = new AdaGrad(graph, learningRate)

  val m = tf.withName("M").variable(tf.constant(0.1f))
  val c1 = tf.variable(tf.constant(0f))
  val c2 = tf.variable(tf.constant(0f))

  val x = tf.withName("X").placeholder(TFloat32.DTYPE)
  val y = tf.withName("Y").placeholder(TFloat32.DTYPE)

  val loss1 = tf.withName("loss1").math.squaredDifference(y, tf.math.add(tf.math.mul(m, x), c1))
  val loss2 = tf.withName("loss2").math.squaredDifference(y, tf.math.add(tf.math.mul(m, x), c2))

  val minimize1 = minimizer(graph, optimizer, loss1, Array(m, c1), "minimize1")

  val minimize2 = minimizer(graph, optimizer, loss2, Array(m, c2), "minimize2")

  lazy val signature = Signature
    .builder()
    .input("X", x)
    .input("Y", y)
    .output("loss1", loss1)
    .output("loss2", loss2)
    .build()

  lazy val asFunc = ConcreteFunction.create(signature, graph)

  def fit(xy: Seq[(Float, Float)]) = Using(new Session(graph)) { session =>
    session.run(tf.init())
    xy.zipWithIndex.foreach { case ((xdata, ydata), j) =>
      val xTensor = TFloat32.tensorOf(StdArrays.ndCopyOf(Array(xdata)))
      val yTensor = TFloat32.tensorOf(StdArrays.ndCopyOf(Array(ydata)))
      val min = if (j % 2 == 0) minimize1 else minimize2
      session
        .runner()
        .feed(x, xTensor)
        .feed(y, yTensor)
        .addTarget(min)
        .run()
    }
    println(
      s"Got m = ${opLookup(m, session)}, c1 = ${opLookup(c1, session)}, c2 = ${opLookup(c2, session)}"
    )
    val concreteFunc = ConcreteFunction.create(
      signature,
      session
    )
    concreteFunc.save("model")
  }
}
