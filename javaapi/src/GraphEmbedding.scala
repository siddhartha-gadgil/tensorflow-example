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
import doodle.image._
// Colors and other useful stuff
import doodle.core._
import scala.collection.immutable.Nil
object DoodleDraw {
  import doodle.image.syntax._
  import doodle.java2d._
  def xyImage(xy: List[(Float, Float)]): Image =
    xy match {
      case Nil => Image.empty
      case head :: next =>
        Image
          .circle(10)
          .fillColor(Color.blue)
          .at(Point(head._1, head._2))
          .on(xyImage(next))
    }

  def xyPlot(xy: List[(Float, Float)]) = xyImage(xy).draw()
}

object GraphEmbedding {
  val rnd = new Random()
  val N = 20
  val linMat = Array.tabulate(N, N) { case (i: Int, j: Int) =>
    if (scala.math.abs(i - j) < 2 || Set(i, j) == Set(0, N - 1)) 1.0f else 0.0f
  }
  def run() = {
    Using(new Graph()) { graph =>
      println("running graph embedding")
      val g = new GraphEmbedding(linMat.size, graph)
      g.fit(linMat)
    }
    Using(new Graph()) { graph =>
      println("fitting in sequence")
      val g = new GraphEmbeddingSeq(linMat.size, graph)
      g.fitSeq(linMat)
    }
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

  val oneMatrix = tf.constant(Array.fill(numPoints, numPoints)(1.0f))

  val oneEpsMatrix =
    tf.constant(Array.fill(numPoints, numPoints)(1.0f + epsilon))

  val probs = tf.math.div(
    oneMatrix,
    tf.math.add(
      oneEpsMatrix,
      totDiff
    )
  )

  val incidence = tf.placeholder(TFloat32.DTYPE)

  val loss = tf.math.neg(
    tf.reduceSum(
      (
        tf.math.add(
          tf.math.mul(incidence, tf.math.log(probs)),
          tf.math.mul(
            tf.math.sub(oneMatrix, incidence),
            tf.math.log(tf.math.sub(oneMatrix, probs))
          )
        )
      ),
      tf.constant(Array(0, 1, 2))
    )
  )

  val optimizer = new Adam(graph)

  val minimize = optimizer.minimize(loss)

  val indicator1 = tf.placeholder(TFloat32.DTYPE)

  val indicator2 = tf.placeholder(TFloat32.DTYPE)

  val x1 = dot(xs, indicator1)
  val y1 = dot(ys, indicator1)
  val x2 = dot(xs, indicator2)
  val y2 = dot(ys, indicator2)

  val dist = tf.math.add(
    tf.math.squaredDifference(x1, x2),
    tf.math.squaredDifference(y1, y2)
  )

  val pSing = tf.math.div(
    tf.constant(1.0f),
    tf.math.add(tf.constant(1.0f + epsilon), dist)
  )

  val qSing = tf.placeholder(TFloat32.DTYPE)

  val lSing = tf.math.neg(
    (
      tf.math.add(
        tf.math.mul(qSing, tf.math.log(pSing)),
        tf.math.mul(
          tf.math.sub(tf.constant(1.0f), qSing),
          tf.math.log(tf.math.sub(tf.constant(1.0f), pSing))
        )
      )
    )
  )

//   val minSing = optimizer.minimize(lSing)

  def dot(v: Operand[TFloat32], w: Operand[TFloat32]) =
    tf.reduceSum(tf.math.mul(v, w), tf.constant(0))

  def fit(inc: Array[Array[Float]], steps: Int = 100000) = {
    Using(new Session(graph)) { session =>
      session.run(tf.init())
      println("initialized")
      val pd = dataLookup(probs, session)
      val pmat = Vector.tabulate(numPoints, numPoints) { case (i, j) =>
        pd.getFloat(i, j)
      }
      val xd = dataLookup(xs, session)
      val yd = dataLookup(ys, session)
      val points =
        (0 until (inc.size)).map(n => (xd.getFloat(n), yd.getFloat(n)))
      println(points.mkString(", "))
      println(pmat.mkString("\n"))
      val incT = TFloat32.tensorOf(StdArrays.ndCopyOf(Array(inc)))
      val lData = session.runner().feed(incidence, incT).fetch(loss).run()
      println(lData)
      val l = lData.get(0).expect(TFloat32.DTYPE).data()
      println(l)
      println(l.getFloat())
      println("Tuning")
      (1 to steps).foreach(_ =>
        session.runner().feed(incidence, incT).addTarget(minimize).run()
      )
      val tundedData = session
        .runner()
        .feed(incidence, incT)
        .fetch(loss)
        .run()
        .get(0)
        .expect(TFloat32.DTYPE)
        .data()
      println(tundedData.getFloat())
      val tpd = dataLookup(probs, session)
      val tpmat = Vector.tabulate(numPoints, numPoints) { case (i, j) =>
        tpd.getFloat(i, j)
      }
      println(tpmat.mkString("\n"))
      val txd = dataLookup(xs, session)
      val tyd = dataLookup(ys, session)
      val tpoints =
        (0 until (inc.size)).map(n =>
          (txd.getFloat(n) * 100f, tyd.getFloat(n) * 100f)
        )
      println(tpoints.mkString(", "))
      tpoints.foreach { case (x, y) => println(s"$x, $y") }
      DoodleDraw.xyPlot(tpoints.toList)
    }
  }

}

class GraphEmbeddingSeq(numPoints: Int, graph: Graph, epsilon: Float = 0.01f) {
  val tf = Ops.create(graph)

  val optimizer = new Adam(graph)

  val xs = tf.variable(
    tf.constant(Array.fill(numPoints)(rnd.nextFloat() * 2.0f))
  )

  val ys = tf.variable(
    tf.constant(Array.fill(numPoints)(rnd.nextFloat() * 2.0f))
  )

  val indicator1 = tf.placeholder(TFloat32.DTYPE)

  val indicator2 = tf.placeholder(TFloat32.DTYPE)

  val x1 = dot(xs, indicator1)
  val y1 = dot(ys, indicator1)
  val x2 = dot(xs, indicator2)
  val y2 = dot(ys, indicator2)

  val dist = tf.math.add(
    tf.math.squaredDifference(x1, x2),
    tf.math.squaredDifference(y1, y2)
  )

  val pSing = tf.math.div(
    tf.constant(1.0f),
    tf.math.add(tf.constant(1.0f + epsilon), dist)
  )

  val qSing = tf.placeholder(TFloat32.DTYPE)

  val lSing = tf.math.neg(
    (
      tf.math.add(
        tf.math.mul(qSing, tf.math.log(pSing)),
        tf.math.mul(
          tf.math.sub(tf.constant(1.0f), qSing),
          tf.math.log(tf.math.sub(tf.constant(1.0f), pSing))
        )
      )
    )
  )

  val minSing = optimizer.minimize(lSing)

  def dot(v: Operand[TFloat32], w: Operand[TFloat32]) =
    tf.reduceSum(tf.math.mul(v, w), tf.constant(0))

  def fitSeq(inc: Array[Array[Float]], steps: Int = 500000) = {
    val N = inc.size
    Using(new Session(graph)) { session =>
      session.run(tf.init())
      println("initialized")
      println("Tuning")
      (1 to steps).foreach { _ =>
        val i1 = rnd.nextInt(N)
        val i2 = rnd.nextInt(N)
        // println(i1)
        val q = inc(i1)(i2)
        val ind1 = Array.tabulate(N)(j => if (j == i1) 1.0f else 0.0f)
        val ind2 = Array.tabulate(N)(j => if (j == i2) 1.0f else 0.0f)
        // println(ind1.toVector)
        session
          .runner()
          .feed(qSing, TFloat32.scalarOf(q))
          .feed(indicator1, TFloat32.vectorOf(ind1: _*))
          .feed(indicator2, TFloat32.vectorOf(ind2: _*))
          .addTarget(minSing)
          .run()
      }
      println("Finished tuning")
      val tundedData = session
        .runner()
        .fetch(xs)
        .fetch(ys)
        .run()
      println(tundedData)
      val txd = tundedData.get(0).expect(TFloat32.DTYPE).data()
      val tyd = tundedData.get(1).expect(TFloat32.DTYPE).data()
      val tpoints =
        (0 until (inc.size)).map(n =>
          (txd.getFloat(n) * 50f, tyd.getFloat(n) * 50f)
        )
      println(tpoints.mkString(", "))
      tpoints.foreach { case (x, y) => println(s"$x, $y") }
      DoodleDraw.xyPlot(tpoints.toList)
    }
  }
}
