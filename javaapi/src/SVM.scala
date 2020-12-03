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
import scala.util.Try

object SVM {
  val dataArr = os.read
    .lines(os.pwd / "javaapi" / "resources" / "linearly_separable_data.csv")
    .map(_.split(",").map(_.toFloat))

  val data = dataArr.map(arr => (2 * arr(0) - 1, (arr(1), arr(2))))

  def run() =
    Using(new Graph()) { graph =>
      println("running SVM")
      val svm = Try(new SVM(graph, 0.1f, 10f)).fold(
        err => {
          println(err)
          err.asInstanceOf[Exception].printStackTrace()
          throw err
        },
        identity(_)
      )
      println("Got svm class")
      svm.fit(data)
    }
}

class SVM(graph: Graph, learningRate: Float, svmC: Float) {
  val tf = Ops.create(graph)
  val b = tf.variable(tf.constant(Array(Array(0.1f))))
  val w = tf.variable(tf.constant(Array(Array(0.1f, 0.7f))))

  val x = tf.withName("X").placeholder(TFloat32.DTYPE)
  val y = tf.withName("Y").placeholder(TFloat32.DTYPE)

  val yRaw = tf.math.add(tf.linalg.matMul(x, w), b)
  val regularizationLoss =
    tf.math.mul(
      tf.constant(0.5f),
      tf.reduceSum(
        tf.math.squaredDifference(w, tf.zerosLike(w)),
        tf.constant(Array(0, 1))
      )
    )

  val hingeLoss = tf.math.maximum(
    tf.constant(0f),
    tf.math.sub(tf.constant(1f), tf.math.mul(y, yRaw))
  )

  val svmLoss =
    tf.math.add(regularizationLoss, tf.math.mul(tf.constant(svmC), hingeLoss))

  val optimizer = new Adam(graph, learningRate)

  val minimize = optimizer.minimize(svmLoss)

  def fit(xy: Seq[(Float, (Float, Float))]) = Using(new Session(graph)) {
    session =>
      session.run(tf.init())
      xy.foreach { case (ydata, (x1, x2)) =>
        // println(s"fitting $ydata, ($x1, $x2)")
        val xTensor =
          TFloat32.tensorOf(StdArrays.ndCopyOf(Array(Array(x1), Array(x2))))
        val yTensor = TFloat32.tensorOf(StdArrays.ndCopyOf(Array(Array(ydata))))
        session
          .runner()
          .feed(x, xTensor)
          .feed(y, yTensor)
          .addTarget(minimize)
          .run()
      }
      println(s"ran fit")
      val bValue = opLookup(b, session)
      println(s"b = $bValue")
      val wValue = {
        val data = dataLookup(w, session)
        Vector(data.getFloat(0, 0), data.getFloat(0, 1))
      }
      println(s"w = $wValue")
    // println(s"Got b = ${opLookup(b, session)} and w = ${opLookup(w, session)}")
  }
}
