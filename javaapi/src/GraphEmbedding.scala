package javaapi

import org.tensorflow._
import org.tensorflow.op._
import org.tensorflow.types._
import scala.util.Using
import org.tensorflow.ndarray._
import org.tensorflow.framework.optimizers._
import org.tensorflow.framework.losses.BinaryCrossentropy

import Utils._
import scala.util._
import GraphEmbedding._

import org.tensorflow.op.core.Assign
import org.tensorflow.op.core.Max

import doodle.reactor._
import doodle.image._
import doodle.java2d._
// Colors and other useful stuff
import scala.collection.immutable.Nil
import doodle.core.Color
object GraphEmbedding {
  val rnd = new Random()
  val N = 75

  var fitDone1 = false

  var fitDone2 = false

  var fitDone3 = false

  var fitDone0 = false

  var stepsRun = 0

  var dataSnap
      : (Vector[(Float, Float)], Vector[((Float, Float), (Float, Float))]) =
    (Vector(), Vector())

  val linMat = Array.tabulate(N, N) { case (i: Int, j: Int) =>
    if (scala.math.abs(i - j) < 5 || (i + N - j < 5) || (j + N - i) < 5) 1.0f
    else 0.0f
  }

  val linMat2 = Array.tabulate(N, N) { case (i: Int, j: Int) =>
    if (scala.math.abs(i - j) < 8 || (i + N - j < 8) || (j + N - i) < 8) 1.0f
    else 0.0f
  }

  @scala.annotation.tailrec
  def getSample(
      remaining: Vector[Int],
      size: Int,
      accum: Vector[Int] = Vector()
  ): Vector[Int] =
    if (size < 1) accum
    else {
      val pick = remaining(rnd.nextInt(remaining.size))
      getSample(remaining.filterNot(_ == pick), size - 1, accum :+ pick)
    }

  def predictRun() = {
    fitDone0 = false
    Using(new Graph()) { graph =>
      println("running prediction based graph embedding")
      val g = Try(new GraphPredictEmbedding(linMat.size, graph)).fold(
        fa => {
          println(fa.getMessage())
          fa.printStackTrace
          throw fa
        },
        identity(_)
      )
      println("created graph")
      println(g.fit(linMat))
    }
  }

  def predictDualRun() = {
    fitDone0 = false
    Using(new Graph()) { graph =>
      println("running multiple predictions based graph embedding")
      val g = Try(new GraphDualPredictEmbedding(linMat.size, graph)).fold(
        fa => {
          println(fa.getMessage())
          fa.printStackTrace
          throw fa
        },
        identity(_)
      )
      println("created graph")
      g.fit(linMat, linMat2)
    }.flatten
  }

  def run() = {
    Using(new Graph()) { graph =>
      println("running graph embedding")
      val g = new GraphEmbedding(linMat.size, graph)
      val animReal =
        Reactor
          .init(())
          .onTick(_ => ())
          .render { (_) =>
            DoodleDraw.showSteps(stepsRun)
            val (points, lines) = dataSnap
            DoodleDraw.linesImage(
              lines.toList,
              DoodleDraw.xyImage(points.toList)
            )
          }
          .stop(_ => fitDone1)
      animReal.run(Frame.size(800, 800))
      g.fit(linMat)
    }

    val animReal =
      Reactor
        .init(())
        .onTick(_ => ())
        .render { (_) =>
          DoodleDraw.showSteps(stepsRun)
          val (points, lines) = dataSnap
          DoodleDraw.linesImage(
            lines.toList,
            DoodleDraw
              .xyImage(points.toList)
              .on(Image.rectangle(1000, 1000).fillColor(Color.black))
          )
        }
        .stop(_ => fitDone0)
    animReal.run(Frame.size(1000, 1000))

    val reps: IndexedSeq[Vector[(Float, Float, Float)]] =
      (1 to 2).map(_ => predictDualRun().get)
    // (1 to 3).foreach(_ => predictRun())
    reps.foreach { repV =>
      val rep = repV.map { case (x, y, z) => Array(x, y, z) }
      val masked = (0 until (N)).filter(_ => rnd.nextDouble() < 0.1).toVector
      @scala.annotation.tailrec
      def unmasked(): Int = {
        val pick = rnd.nextInt(N)
        if (masked.contains(pick)) unmasked() else pick
      }

      def normGraph = {
        val j = unmasked()
        (rep(j), j.toFloat)
      }

      def queries = masked.map(rep(_))

      Using(new Graph()) { graph =>
        println("fitting via representation")
        val funcAp = new FunctionApproximatorMultiDim(graph, 3, 20)
        val fitted = funcAp
          .fit(normGraph, 100000, queries)
          .fold(
            fa => {
              println(fa.getMessage())
              println(fa.getStackTrace())
              throw fa
            },
            identity(_)
          )
        println(fitted.zip(masked).mkString("\n"))
      }
    }

    /*
    Using(new Graph()) { graph =>
      println("running batched graph embedding")
      val g = Try(new GraphEmbeddingBatched(linMat.size, 20, graph)).fold(
        fa => {
          println(fa.getMessage())
          println(fa.getStackTrace())
          throw fa
        },
        identity(_)
      )
      println("starting animation")
      val animReal =
        Reactor
          .init(())
          .onTick(_ => ())
          .render { (_) =>
            DoodleDraw.showSteps(stepsRun)
            val (points, lines) = dataSnap
            DoodleDraw.linesImage(
              lines.toList,
              DoodleDraw.xyImage(points.toList)
            )
          }
          .stop(_ => fitDone2)
      animReal.run(Frame.size(800, 800))
      g.fit(linMat)
    }

    Using(new Graph()) { graph =>
      println("fitting in sequence")
      val g = new GraphEmbeddingSeq(linMat.size, graph)
      val animReal =
        Reactor
          .init(())
          .render { (_) =>
            DoodleDraw.showSteps(stepsRun)
            val (points, lines) = dataSnap
            DoodleDraw.linesImage(
              lines.toList,
              DoodleDraw.xyImage(points.toList)
            )
          }
          .stop(_ => fitDone3)
      animReal.run(Frame.size(800, 800))
      g.fitSeq(linMat)
    }
     */
  }

  def pointsAndLines(txs: TFloat32, tys: TFloat32, n: Int) = {
    val tpoints =
      (0 until n)
        .map(n => (txs.getFloat(n) * 50f, tys.getFloat(n) * 50f))
        .toVector
    val tLines = tpoints.zip(tpoints.tail :+ tpoints.head)
    (tpoints, tLines)
  }

  import scala.math.{sin, cos}
  def project(
      theta: Double,
      phi: Double,
      L: Double
  )(x: Float, y: Float, z: Float): (Float, Float) = {
    import scala.math.{sin, cos}
    val zhat = x * sin(theta) - y * cos(theta)
    val scale = L / (L - zhat)
    (
      ((x * cos(theta) * sin(phi) + y * sin(theta) * sin(phi) + z * cos(
        phi
      )) * scale).toFloat,
      ((x * cos(theta) * cos(phi) + y * sin(theta) * cos(phi) - z * sin(
        phi
      )) * scale).toFloat
    )
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

  val incidence = tf.placeholder(classOf[TFloat32])

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
      tf.constant(Array(0, 1))
    )
  )

  val optimizer = new Adam(graph)

  val minimize = optimizer.minimize(loss)

  def fit(inc: Array[Array[Float]], steps: Int = 40000) = {
    Using(new Session(graph)) { session =>
      session.run(tf.init())
      println("initialized")
      val incT = TFloat32.tensorOf(StdArrays.ndCopyOf(inc))
      println("Tuning")
      (1 to steps).foreach { j =>
        val tData = session
          .runner()
          .feed(incidence, incT)
          .addTarget(minimize)
          .fetch(xs)
          .fetch(ys)
          .run()
        val xd = tData.get(0).asInstanceOf[TFloat32]
        val yd = tData.get(1).asInstanceOf[TFloat32]
        val unscaledPoints: Vector[(Float, Float)] =
          (0 until (inc.size))
            .map(n => (xd.getFloat(n), yd.getFloat(n)))
            .toVector
        val maxX = unscaledPoints.map(_._1).max
        val maxY = unscaledPoints.map(_._2).max
        val scale = scala.math.min(300f / maxX, 300f / maxY)
        val points = unscaledPoints.map { case (x, y) =>
          (x * scale, y * scale)
        }
        val lines = points.zip(points.tail :+ points.head)
        stepsRun = j
        dataSnap = (points, lines)
      // (points, lines)
      }
      fitDone1 = true
      println("Tuning complete")
      val tundedData = session
        .runner()
        .feed(incidence, incT)
        .fetch(loss)
        .run()
        .get(0)
        .asInstanceOf[TFloat32]

      println(tundedData.getFloat())
      val txd = dataLookup(xs, session)
      val tyd = dataLookup(ys, session)
      val tpoints =
        (0 until (inc.size)).map(n =>
          (txd.getFloat(n) * 60f, tyd.getFloat(n) * 60f)
        )
    }
  }

}

class GraphPredictEmbedding(
    numPoints: Int,
    graph: Graph,
    epsilon: Float = 0.01f
) {
  val tf = Ops.create(graph)

  val ones = tf.constant(Array.fill(numPoints)(1.0f))

  val xs = tf.variable(
    tf.constant(Array.fill(numPoints)(rnd.nextFloat() * 2.0f))
  )

  val ys = tf.variable(
    tf.constant(Array.fill(numPoints)(rnd.nextFloat() * 2.0f))
  )

  val zs = tf.variable(
    tf.constant(Array.fill(numPoints)(rnd.nextFloat() * 2.0f))
  )

  val repXs = tf.variable(
    tf.constant(Array.fill(numPoints)(rnd.nextFloat() * 2.0f))
  )

  val repYs = tf.variable(
    tf.constant(Array.fill(numPoints)(rnd.nextFloat() * 2.0f))
  )

  val repZs = tf.variable(
    tf.constant(Array.fill(numPoints)(rnd.nextFloat() * 2.0f))
  )

  def rankOne(v: Operand[TFloat32], w: Operand[TFloat32]) =
    tf.linalg.matMul(
      tf.reshape(v, tf.constant(Array(numPoints, 1))),
      tf.reshape(w, tf.constant(Array(1, numPoints)))
    )

  val xProds = tf.math.mul(rankOne(xs, ones), rankOne(ones, repXs))

  val yProds = tf.math.mul(rankOne(ys, ones), rankOne(ones, repYs))

  val zProds = tf.math.mul(rankOne(zs, ones), rankOne(ones, repZs))

  val dotProds = tf.math.add(zProds, tf.math.add(xProds, yProds))

  val oneMatrix = tf.constant(Array.fill(numPoints, numPoints)(1.0f))

  val oneEpsMatrix =
    tf.constant(Array.fill(numPoints, numPoints)(1.0f + epsilon))

  val probs = tf.math.sigmoid(dotProds)

  val incidence = tf.placeholder(classOf[TFloat32])

  val bce = new BinaryCrossentropy(tf)

  val loss = // bce.call(incidence, probs)
    tf.math.neg(
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
        tf.constant(Array(0, 1))
      )
    )

  // max(x, 0) - x * z + log(1 + exp(-abs(x)))
  val stableLoss = tf.math.add(
    tf.math
      .sub(
        tf.math.maximum(dotProds, tf.constant(0f)),
        tf.math.mul(dotProds, incidence)
      ),
    tf.math.log(
      tf.math
        .add(tf.constant(1f), tf.math.exp(tf.math.neg(tf.math.abs(dotProds))))
    )
  )

  val optimizer = new Adam(graph)

  val minimize = optimizer.minimize(stableLoss)

  def fit(inc: Array[Array[Float]], steps: Int = 200000) = {
    Using(new Session(graph)) { session =>
      session.run(tf.init())
      println("initialized")
      val incT = TFloat32.tensorOf(StdArrays.ndCopyOf(inc))
      println("Tuning")
      (1 to steps).foreach { j =>
        val tData = session
          .runner()
          .feed(incidence, incT)
          .addTarget(minimize)
          .fetch(xs)
          .fetch(ys)
          .fetch(zs)
          .run()
        val xd = tData.get(0).asInstanceOf[TFloat32]
        val yd = tData.get(1).asInstanceOf[TFloat32]
        val zd: TFloat32 = tData.get(2).asInstanceOf[TFloat32]
        import scala.math.sqrt
        def zoom(a: Float) =
          a // if (a >= 0) sqrt(a).toFloat else -sqrt(-a).toFloat
        val base3dPoints: Vector[(Float, Float, Float)] =
          (0 until (inc.size))
            .map(n =>
              (zoom(xd.getFloat(n)), zoom(yd.getFloat(n)), zoom(zd.getFloat(n)))
            )
            .toVector
        val xavg = base3dPoints.map(_._1).sum / base3dPoints.size
        val yavg = base3dPoints.map(_._2).sum / base3dPoints.size
        val zavg = base3dPoints.map(_._3).sum / base3dPoints.size
        val unscaled3dPoints = base3dPoints.map { case (x, y, z) =>
          (x - xavg, y - yavg, z - zavg)
        }
        val theta = j.toDouble / 3000
        val phi = j.toDouble / 4231
        val maxX = unscaled3dPoints.map(_._1.abs).max
        val maxY = unscaled3dPoints.map(_._2.abs).max
        val maxZ = unscaled3dPoints.map(_._3.abs).max
        val scale = List(300f / maxX, 300f / maxY, 300f / maxZ).min
        val scaled3dPoints = unscaled3dPoints.map { case (x, y, z) =>
          (x * scale, y * scale, z * scale)
        }
        val points: Vector[(Float, Float)] = scaled3dPoints.map {
          case (x, y, z) => project(theta, phi, 3000)(x, y, z)
        }
        // val points = unscaledPoints.map { case (x, y) =>
        //   (x * scale, y * scale)
        // }
        val lines = points.zip(points.tail :+ points.head)
        stepsRun = j
        dataSnap = (points, lines)
      }
      fitDone0 = true
      println("Tuning complete")
      val tData = session
        .runner()
        .feed(incidence, incT)
        .addTarget(minimize)
        .fetch(xs)
        .fetch(ys)
        .fetch(zs)
        .run()
      val xd = tData.get(0).asInstanceOf[TFloat32]
      val yd = tData.get(1).asInstanceOf[TFloat32]
      val zd: TFloat32 = tData.get(2).asInstanceOf[TFloat32]
      import scala.math.sqrt

      (0 until (inc.size))
        .map(n => (xd.getFloat(n), yd.getFloat(n), zd.getFloat(n)))
        .toVector

    }
  }

}

class GraphDualPredictEmbedding(
    numPoints: Int,
    graph: Graph,
    epsilon: Float = 0.01f
) {
  val tf = Ops.create(graph)

  val ones = tf.constant(Array.fill(numPoints)(1.0f))

  val xs = tf.variable(
    tf.constant(Array.fill(numPoints)(rnd.nextFloat() * 2.0f))
  )

  val ys = tf.variable(
    tf.constant(Array.fill(numPoints)(rnd.nextFloat() * 2.0f))
  )

  val zs = tf.variable(
    tf.constant(Array.fill(numPoints)(rnd.nextFloat() * 2.0f))
  )

  val repXs1 = tf.variable(
    tf.constant(Array.fill(numPoints)(rnd.nextFloat() * 2.0f))
  )

  val repYs1 = tf.variable(
    tf.constant(Array.fill(numPoints)(rnd.nextFloat() * 2.0f))
  )

  val repZs1 = tf.variable(
    tf.constant(Array.fill(numPoints)(rnd.nextFloat() * 2.0f))
  )

  val repXs2 = tf.variable(
    tf.constant(Array.fill(numPoints)(rnd.nextFloat() * 2.0f))
  )

  val repYs2 = tf.variable(
    tf.constant(Array.fill(numPoints)(rnd.nextFloat() * 2.0f))
  )

  val repZs2 = tf.variable(
    tf.constant(Array.fill(numPoints)(rnd.nextFloat() * 2.0f))
  )

  def rankOne(v: Operand[TFloat32], w: Operand[TFloat32]) =
    tf.linalg.matMul(
      tf.reshape(v, tf.constant(Array(numPoints, 1))),
      tf.reshape(w, tf.constant(Array(1, numPoints)))
    )

  val xProds1 = tf.math.mul(rankOne(xs, ones), rankOne(ones, repXs1))

  val yProds1 = tf.math.mul(rankOne(ys, ones), rankOne(ones, repYs1))

  val zProds1 = tf.math.mul(rankOne(zs, ones), rankOne(ones, repZs1))

  val dotProds1 = tf.math.add(zProds1, tf.math.add(xProds1, yProds1))

  val xProds2 = tf.math.mul(rankOne(xs, ones), rankOne(ones, repXs2))

  val yProds2 = tf.math.mul(rankOne(ys, ones), rankOne(ones, repYs2))

  val zProds2 = tf.math.mul(rankOne(zs, ones), rankOne(ones, repZs2))

  val dotProds2 = tf.math.add(zProds2, tf.math.add(xProds2, yProds2))

  val oneMatrix = tf.constant(Array.fill(numPoints, numPoints)(1.0f))

  val oneEpsMatrix =
    tf.constant(Array.fill(numPoints, numPoints)(1.0f + epsilon))

  val probs1 = tf.math.sigmoid(dotProds1)

  val incidence1 = tf.placeholder(classOf[TFloat32])

  val probs2 = tf.math.sigmoid(dotProds2)

  val incidence2 = tf.placeholder(classOf[TFloat32])

  // val loss = // bce.call(incidence, probs)
  //   tf.math.neg(
  //     tf.reduceSum(
  //       (
  //         tf.math.add(
  //           tf.math.mul(incidence, tf.math.log(probs)),
  //           tf.math.mul(
  //             tf.math.sub(oneMatrix, incidence),
  //             tf.math.log(tf.math.sub(oneMatrix, probs))
  //           )
  //         )
  //       ),
  //       tf.constant(Array(0, 1))
  //     )
  //   )

  // max(x, 0) - x * z + log(1 + exp(-abs(x)))
  val stableLoss1 = tf.math.add(
    tf.math
      .sub(
        tf.math.maximum(dotProds1, tf.constant(0f)),
        tf.math.mul(dotProds1, incidence1)
      ),
    tf.math.log(
      tf.math
        .add(tf.constant(1f), tf.math.exp(tf.math.neg(tf.math.abs(dotProds1))))
    )
  )

  val stableLoss2 = tf.math.add(
    tf.math
      .sub(
        tf.math.maximum(dotProds2, tf.constant(0f)),
        tf.math.mul(dotProds2, incidence2)
      ),
    tf.math.log(
      tf.math
        .add(tf.constant(1f), tf.math.exp(tf.math.neg(tf.math.abs(dotProds2))))
    )
  )

  val stableLoss = tf.math.add(stableLoss1, stableLoss2)

  val optimizer = new Adam(graph)

  val minimize = optimizer.minimize(stableLoss)

  def fit(
      inc1: Array[Array[Float]],
      inc2: Array[Array[Float]],
      steps: Int = 200000
  ) = {
    Using(new Session(graph)) { session =>
      session.run(tf.init())
      println("initialized")
      val inc1T = TFloat32.tensorOf(StdArrays.ndCopyOf(inc1))
      val inc2T = TFloat32.tensorOf(StdArrays.ndCopyOf(inc2))
      println("Tuning")
      (1 to steps).foreach { j =>
        val tData = session
          .runner()
          .feed(incidence1, inc1T)
          .feed(incidence2, inc2T)
          .addTarget(minimize)
          .fetch(xs)
          .fetch(ys)
          .fetch(zs)
          .run()
        val xd = tData.get(0).asInstanceOf[TFloat32]
        val yd = tData.get(1).asInstanceOf[TFloat32]
        val zd: TFloat32 = tData.get(2).asInstanceOf[TFloat32]
        import scala.math.sqrt
        def zoom(a: Float) =
          a // if (a >= 0) sqrt(a).toFloat else -sqrt(-a).toFloat
        val base3dPoints: Vector[(Float, Float, Float)] =
          (0 until (inc1.size))
            .map(n =>
              (zoom(xd.getFloat(n)), zoom(yd.getFloat(n)), zoom(zd.getFloat(n)))
            )
            .toVector
        val xavg = base3dPoints.map(_._1).sum / base3dPoints.size
        val yavg = base3dPoints.map(_._2).sum / base3dPoints.size
        val zavg = base3dPoints.map(_._3).sum / base3dPoints.size
        val unscaled3dPoints = base3dPoints.map { case (x, y, z) =>
          (x - xavg, y - yavg, z - zavg)
        }
        val theta = j.toDouble / 3000
        val phi = j.toDouble / 4231
        val maxX = unscaled3dPoints.map(_._1.abs).max
        val maxY = unscaled3dPoints.map(_._2.abs).max
        val maxZ = unscaled3dPoints.map(_._3.abs).max
        val scale = List(300f / maxX, 300f / maxY, 300f / maxZ).min
        val scaled3dPoints = unscaled3dPoints.map { case (x, y, z) =>
          (x * scale, y * scale, z * scale)
        }
        val points: Vector[(Float, Float)] = scaled3dPoints.map {
          case (x, y, z) => project(theta, phi, 3000)(x, y, z)
        }
        // val points = unscaledPoints.map { case (x, y) =>
        //   (x * scale, y * scale)
        // }
        val lines = points.zip(points.tail :+ points.head)
        stepsRun = j
        dataSnap = (points, lines)
      }
      fitDone0 = true
      println("Tuning complete")
      val tData = session
        .runner()
        .feed(incidence1, inc1T)
        .feed(incidence2, inc2T)
        .fetch(xs)
        .fetch(ys)
        .fetch(zs)
        .run()
      val xd = tData.get(0).asInstanceOf[TFloat32]
      val yd = tData.get(1).asInstanceOf[TFloat32]
      val zd: TFloat32 = tData.get(2).asInstanceOf[TFloat32]
      import scala.math.sqrt

      (0 until (inc1.size))
        .map(n => (xd.getFloat(n), yd.getFloat(n), zd.getFloat(n)))
        .toVector
    }
  }

}

class GraphEmbeddingBatched(
    numPoints: Int,
    batchSize: Int,
    graph: Graph,
    epsilon: Float = 0.01f
) {
  val tf = Ops.create(graph)

  println("graph batch created")

  val ones = tf.constant(Array.fill(batchSize)(1.0f))

  val fxs = tf.variable(
    tf.constant(Array.fill(numPoints)(rnd.nextFloat() * 2.0f))
  )

  val fys = tf.variable(
    tf.constant(Array.fill(numPoints)(rnd.nextFloat() * 2.0f))
  )

  val maxX: Max[TFloat32] = tf.max(tf.math.abs(fxs), tf.constant(Array(0)))

  val maxY: Max[TFloat32] = tf.max(tf.math.abs(fys), tf.constant(Array(0)))

  val maxCoordScale =
    tf.math.div(tf.math.maximum(maxX, maxY), tf.constant(2000f))

  val rescaleX: Assign[TFloat32] =
    tf.assign(fxs, tf.math.div(fxs, maxCoordScale))

  val rescaleY: Assign[TFloat32] =
    tf.assign(fys, tf.math.div(fys, maxCoordScale))

  val projection = tf.placeholderWithDefault(
    tf.constant(Array.fill(batchSize)(Array.fill(numPoints)(1f))),
    Shape.of(batchSize, numPoints)
  )

  val xs = {
    tf.linalg.matMul(
      projection,
      tf.reshape(fxs, tf.constant(Array(numPoints, 1)))
    )
  }

  val ys = tf.linalg.matMul(
    projection,
    tf.reshape(fys, tf.constant(Array(numPoints, 1)))
  )

  def rankOne(v: Operand[TFloat32], w: Operand[TFloat32]) = {
    // println(
    //   s"calling rank one with ${v.asOutput().shape()} and ${w.asOutput().shape}"
    // )
    val row = tf.reshape(v, tf.constant(Array(batchSize, 1)))
    // println("obtained row")
    tf.linalg.matMul(
      row,
      tf.reshape(w, tf.constant(Array(1, batchSize)))
    )
  }

  val xDiff = tf.math.squaredDifference(rankOne(xs, ones), rankOne(ones, xs))

  val yDiff = tf.math.squaredDifference(rankOne(ys, ones), rankOne(ones, ys))

  val totDiff = tf.math.add(xDiff, yDiff)

  val oneMatrix = tf.constant(Array.fill(batchSize, batchSize)(1.0f))

  val oneEpsMatrix =
    tf.constant(Array.fill(batchSize, batchSize)(1.0f + epsilon))

  val probs = tf.math.div(
    oneMatrix,
    tf.math.add(
      oneEpsMatrix,
      totDiff
    )
  )

  val incidence = tf.placeholder(classOf[TFloat32])

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
      tf.constant(Array(0, 1))
    )
  )

  val optimizer = new Adam(graph)

  val minimize = optimizer.minimize(loss)

  def fit(
      inc: Array[Array[Float]],
      rescale: Boolean = false,
      steps: Int = 2000000
  ) = {
    Using(new Session(graph)) { session =>
      session.run(tf.init())
      println("initialized")
      println("Tuning")
      (1 to steps).foreach { j =>
        val batch = getSample((0 until (numPoints)).toVector, batchSize).toArray
        val incB = Array.tabulate(batchSize)(i =>
          Array.tabulate(batchSize)(j => inc(batch(i))(batch(j)))
        )
        val incT = TFloat32.tensorOf(
          StdArrays.ndCopyOf(
            incB
          )
        )
        val projMat =
          Array.tabulate(batchSize) { i =>
            Array.tabulate(numPoints)(j => if (j == batch(i)) 1f else 0f)
          }
        val projT = TFloat32.tensorOf(
          StdArrays.ndCopyOf(
            projMat
          )
        )
        (if (rescale && j % 5000 == 0)
           Try {
             session.runner().addTarget(rescaleX).addTarget(rescaleY).run()
           }.fold(
             fa => {
               println(fa.getMessage())
               println(fa.printStackTrace())
               throw fa
             },
             identity(_)
           ))
        // println(incB.map(_.toVector).toVector)
        val tData = Try(
          session
            .runner()
            .feed(incidence, incT)
            .feed(projection, projT)
            .addTarget(minimize)
            .fetch(fxs)
            .fetch(fys)
            .run()
        ).fold(
          fa => {
            println(fa.getMessage())
            println(fa.printStackTrace())
            throw fa
          },
          identity(_)
        )
        // println(tData)
        val xd = tData.get(0).asInstanceOf[TFloat32]
        val yd = tData.get(1).asInstanceOf[TFloat32]
        val unscaledPoints: Vector[(Float, Float)] =
          (0 until (inc.size))
            .map(n => (xd.getFloat(n), yd.getFloat(n)))
            .toVector
        val maxX = unscaledPoints.map(_._1).max
        val maxY = unscaledPoints.map(_._2).max
        val scale = scala.math.min(300f / maxX, 300f / maxY)
        val points = unscaledPoints.map { case (x, y) =>
          (x * scale, y * scale)
        }
        val lines = points.zip(points.tail :+ points.head)
        stepsRun = j
        dataSnap = (points, lines)
      // (points, lines)
      }
      fitDone2 = true
      println("Tuning complete")
      val txd = dataLookup(xs, session)
      val tyd = dataLookup(ys, session)
      val tpoints =
        (0 until (inc.size)).map(n =>
          (txd.getFloat(n) * 60f, tyd.getFloat(n) * 60f)
        )
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

  val indicator1 = tf.placeholder(classOf[TFloat32])

  val indicator2 = tf.placeholder(classOf[TFloat32])

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

  val qSing = tf.placeholder(classOf[TFloat32])

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

  def fitSeq(inc: Array[Array[Float]], steps: Int = 2000000) = {
    val N = inc.size
    Using(new Session(graph)) { session =>
      session.run(tf.init())
      println("initialized")
      println("Tuning")
      (1 to steps).foreach { j =>
        val i1 = rnd.nextInt(N)
        val i2 = rnd.nextInt(N)
        val q = inc(i1)(i2)
        val ind1 = Array.tabulate(N)(j => if (j == i1) 1.0f else 0.0f)
        val ind2 = Array.tabulate(N)(j => if (j == i2) 1.0f else 0.0f)
        val tData = session
          .runner()
          .feed(qSing, TFloat32.scalarOf(q))
          .feed(indicator1, TFloat32.vectorOf(ind1: _*))
          .feed(indicator2, TFloat32.vectorOf(ind2: _*))
          .addTarget(minSing)
          .fetch(xs)
          .fetch(ys)
          .run()
        val xd = tData.get(0).asInstanceOf[TFloat32]
        val yd = tData.get(1).asInstanceOf[TFloat32]
        val unscaledPoints: Vector[(Float, Float)] =
          (0 until (inc.size))
            .map(n => (xd.getFloat(n), yd.getFloat(n)))
            .toVector
        val maxX = unscaledPoints.map(_._1).max
        val maxY = unscaledPoints.map(_._2).max
        val scale = scala.math.min(300f / maxX, 300f / maxY)
        val points = unscaledPoints.map { case (x, y) =>
          (x * scale, y * scale)
        }
        val lines = points.zip(points.tail :+ points.head)
        stepsRun = j
        dataSnap = (points, lines)
      }
      fitDone3 = true
      println("Finished tuning")
    // val tundedData = session
    //   .runner()
    //   .fetch(xs)
    //   .fetch(ys)
    //   .run()
    // val txd = tundedData.get(0).asInstanceOf[TFloat32]
    // val tyd = tundedData.get(1).asInstanceOf[TFloat32]
    // val tpoints =
    //   (0 until (inc.size))
    //     .map(n => (txd.getFloat(n) * 50f, tyd.getFloat(n) * 50f))
    //     .toVector
    // val tLines = tpoints.zip(tpoints.tail :+ tpoints.head)
    // DoodleDraw.linesPlot(tpoints.toList, tLines.toList)
    }
  }
}
