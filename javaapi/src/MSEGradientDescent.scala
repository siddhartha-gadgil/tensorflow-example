package javaapi

import org.tensorflow._
import org.tensorflow.op._, core.Variable
import org.tensorflow.types._
import scala.util.Using
import org.tensorflow.framework.optimizers.{Optimizer, GradientDescent}

object MSEGradientDescent {
  val var0Init = Array(1.0f, 2.0f)
  val var1Init = Array(3.0f, 4.0f)

  val learningRate = 0.3f

  def run(): Unit = {
    Using.Manager { use =>
      val graph: Graph = use(new Graph())
      val tf: Ops = Ops.create(graph)

      val var0: Variable[TFloat32] = tf.variable(tf.constant(var0Init))
      val var1: Variable[TFloat32] = tf.variable(tf.constant(var1Init))

      val instance = new GradientDescent(graph, learningRate)

      val loss =
        tf.reduceSum(tf.math.squaredDifference(var0, var1), tf.constant(0))

      val minimize = instance.minimize(loss)

      val session: Session = use(new Session(graph))

      /* initialize the local variables */
      session.run(tf.init())

      def varLookup(v: Variable[TFloat32], sess: Session): Vector[Float] = {
        val result = sess.runner().fetch(v).run()
        val data = result.get(0).asInstanceOf[TFloat32]
        (0 until (data.size().toInt)).toVector.map(n => data.getFloat(n))
      }

      (0 to 20).foreach { n =>
        println(
          s"ran minimize $n times, fetched var0 as ${varLookup(var0, session)}  and var1 as ${varLookup(var1, session)} "
        )
        session.run(minimize)

      }
    }
  }
}
