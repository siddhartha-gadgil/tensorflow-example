package javaapi

import org.tensorflow._
import org.tensorflow.op._
import org.tensorflow.op.core.Placeholder
import org.tensorflow.op.math.Add
import org.tensorflow.types._
import scala.util.Using
import org.tensorflow.ndarray._
import org.tensorflow.framework.optimizers.{Optimizer, GradientDescent}
import scala.jdk.CollectionConverters._
import org.tensorflow.types.family.TType
import Optimizer.GradAndVar

object MSEGradientDescent {
  val var0Init = Array(1.0f, 2.0f)
  val var1Init = Array(3.0f, 4.0f)

  val learningRate = 0.3f

  def run(): Unit = {
    Using.Manager { use =>
      val graph: Graph = use(new Graph())
      val tf : Ops = Ops.create(graph)

      val var0 = tf.variable(tf.constant(var0Init))
      val var1 = tf.variable(tf.constant(var1Init))

      val instance = new GradientDescent(graph, learningRate)

      val loss = tf.math.squaredDifference(var0, var1)
      val minimize = instance.minimize(loss)

      val session = use(new Session(graph))

      /* initialize the local variables */
      session.run(tf.init())

      val result0 = session.runner().fetch(var0).run();
      def res0(n: Int) =
        result0.get(0).expect(TFloat32.DTYPE).data().getFloat(n)
      val result1 = session.runner().fetch(var1).run();
      def res1(n: Int) =
        result1.get(0).expect(TFloat32.DTYPE).data().getFloat(n)

      (0 to 20).foreach { n =>
        println(
          s"ran minimize $n times, fetched var0 as ${res0(0)} and ${res0(
            1
          )} and var1 as ${res1(0)} and ${res1(1)}"
        )
        session.run(minimize)

      }

    }
  }
}
