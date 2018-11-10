package scalaapi

import org.platanios.tensorflow.api
import org.platanios.tensorflow.api._
import Geom._

case class Geom(n: Int, p: Double) {
  val probVars: Vector[api.tf.Variable[Float]] = (0 to n)
    .map(j =>
      tf.variable[Float](s"p$j",
                         Shape(1, 1),
                         initializer = tf.ConstantInitializer[Float](1.0f / n)))
    .toVector

  val totSig: Output[Float] =
    probVars.map(v => tf.sigmoid(v: Output[Float])).reduce[Output[Float]] {
      case (x, y) => tf.add(x, y)
    }

  val probs: Vector[Output[Float]] =
    probVars.map(v => tf.divide(tf.sigmoid(v), totSig))


  val matchErrors: Vector[Output[Float]] =
    probs.zip(probs.tail).map {
      case (x0, x1) =>
        delSq(x1, tf.multiply(x0, p.toFloat))
    }


  val loss: Output[Float] = matchErrors.reduce[Output[Float]] {
    case (a, b) => tf.add(a, b)
  }

  val trainOp: UntypedOp = tf.train.AdaGrad(1.0f).minimize(loss)

  val session = Session()

  session.run(targets = tf.globalVariablesInitializer())

  def tuned(steps: Int): Vector[Float] = {

    (1 to steps).foreach { j =>
//      println(j)
      val trainLoss = session.run(fetches = loss, targets = trainOp)
      if (j % 100 == 0) println(s"loss: ${trainLoss.scalar}, steps: $j")
    }

    (0 to n).toVector.map { j =>
      session.run(fetches = probs(j)).scalar
    }
  }

}

object Geom {
  def delSq(x: Output[Float], y: Output[Float]): Output[Float] =
    tf.square(
      tf.divide(tf.subtract(x, y),
                tf.add(tf.abs(x), tf.abs(y)))
    )
}
