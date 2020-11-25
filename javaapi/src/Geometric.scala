package javaapi

import org.tensorflow._
import org.tensorflow.op._, core.Variable
import org.tensorflow.op.core.Placeholder
import org.tensorflow.op.math.Add
import org.tensorflow.types._
import scala.util.Using
import org.tensorflow.ndarray._
import org.tensorflow.framework.optimizers.{Optimizer, GradientDescent}
import scala.jdk.CollectionConverters._
import GeometricSimple.opLookup
import org.tensorflow.framework.optimizers.AdaGrad

case class GeometricSimple(
    n: Int,
    p: Float,
    graph: Graph,
    learningRate: Float
) {
  val tf = Ops.create(graph)
  val xs = (0 until (n)).toVector.map(j =>
    tf.withName(s"x$j").variable(tf.constant(0f))
  )
  val ps: Vector[Operand[TFloat32]] = xs.map(x => //x
    tf.math.sigmoid(x)
  )
  val total = ps.reduce[Operand[TFloat32]](tf.math.add(_, _))
  val totalError = tf.math.squaredDifference(tf.math.log(total), tf.constant(0.0f))
  val matchErrors: Vector[Operand[TFloat32]] =
    (0 until (n - 1)).toVector.map { j =>
      tf.math.squaredDifference(
        tf.math.log(tf.math.mul(ps(j), tf.constant(p))),
        tf.math.log(ps(j + 1))
      )
    }
  val totalMatchError: Operand[TFloat32] = matchErrors.reduce(tf.math.add(_, _))
  val loss = tf.math.add(totalError, totalMatchError)
  val instance = new GradientDescent(graph, learningRate)
  val minimize = instance.minimize(loss)

  def runSession(): Unit = Using(new Session(graph)) { session =>
    session.run(tf.init())
    (0 to 20000).foreach { n =>
      if (n % 1000 == 0)
        println(
          s"ran minimize $n times, fetched  ${ps.map(p => opLookup(p, session))}, total ${opLookup(total, session)} "
        )
      session.run(minimize)
    }
  }

}

object GeometricSimple {
  val n = 20
  val p = 0.5f
  val learningRate: Float = 0.03f

  def run(): Unit = {
    Using(new Graph()) { graph =>
      GeometricSimple(n, p, graph, learningRate).runSession()
    }
  }

  def opLookup(v: Operand[TFloat32], sess: Session): Float = {
    val result = sess.runner().fetch(v).run()
    val data = result.get(0).expect(TFloat32.DTYPE).data()
    data.getFloat()
  }
}
