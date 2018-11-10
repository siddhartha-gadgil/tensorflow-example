package scalaapi

import org.platanios.tensorflow.api
import org.platanios.tensorflow.api._

import SimpleGeom._

case class SimpleGeom(n: Int, p: Double) {
  val probVars: Vector[api.tf.Variable[Float]] = (0 to n)
    .map(j =>
      tf.variable[Float](s"p$j", Shape(1, 1), initializer = tf.ConstantInitializer[Float](1.0f / n)))
    .toVector

  val totProbErr: Output[Float] =
    tf.square(probVars.foldLeft[Output[Float]](tf.constant[Float](-1)) {
      case (x, y) => tf.add(x, y)
    })

  val matchErrors: Vector[Output[Float]] =
    probVars.zip(probVars.tail).map {
      case (x0, x1) =>
        delSq(x1, tf.multiply(x0, p.toFloat))
    }

  val posErrs: Vector[Output[Float]] =
    probVars.map(v => tf.square(tf.multiply(tf.subtract(v, tf.abs(v)), 100f)))

  val loss: Output[Float] = (matchErrors ++ posErrs).foldLeft(totProbErr) {
    case (a, b) => tf.add(a, b)
  }

  val trainOp: UntypedOp = tf.train.AdaGrad(0.1f).minimize(loss)

  val session = Session()

  session.run(targets = tf.globalVariablesInitializer())

  def tuned(steps: Int): Vector[Float] = {

    (1 to steps).foreach { j =>
//      println(j)
      val trainLoss = session.run(fetches = loss, targets = trainOp)
      if (j % 100 == 0) println(trainLoss.scalar)
    }

    (1 to n).toVector.map { j =>
      session.run(fetches = probVars(j).value).scalar
    }
  }

}

object SimpleGeom {
  def delSq(x: Output[Float], y: Output[Float]): Output[Float] =
    tf.square(
      tf.divide(tf.subtract(x, y),
                tf.add(tf.add(tf.abs(x), tf.abs(y)), math.pow(10, -7).toFloat))
    )
}
