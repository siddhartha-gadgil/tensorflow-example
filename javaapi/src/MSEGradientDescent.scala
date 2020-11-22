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

  val shape0 = Shape.of(var0Init.length)
  val shape1 = Shape.of(var1Init.length)

  def run(): Unit = {
    Using.Manager { use =>
      val graph = use(new Graph())
      val session = use(new Session(graph))
      val tf = Ops.create(graph)

      val var0 = tf.withName("var0").variable(shape0, TFloat32.DTYPE)
      val var1 = tf.withName("var1").variable(shape1, TFloat32.DTYPE)

      val var0Initializer = tf.assign(var0, tf.constant(var0Init))
      val var1Initializer = tf.assign(var1, tf.constant(var1Init))

      val instance = new GradientDescent(graph, learningRate)

      val loss = tf.math.squaredDifference(var0, var1)
      val minimize = instance.minimize(loss)

      /* initialize the local variables */
      session.run(var0Initializer);
      session.run(var1Initializer);

      //   session.run(update)

      /* initialize the accumulators */
      session.runner().addTarget(tf.init()).run()

      val result0 = session.runner().fetch("var0").run();
      def res0(n: Int) =
        result0.get(0).expect(TFloat32.DTYPE).data().getFloat(n)
      val result1 = session.runner().fetch("var1").run();
      def res1(n: Int) =
        result1.get(0).expect(TFloat32.DTYPE).data().getFloat(n)

      (1 to 10).foreach { _ =>
        session.run(minimize)
        println(
          s"ran minimize, fetched var0 as ${res0(0)} and ${res0(
            1
          )} and var1 as ${res1(0)} and ${res1(1)}"
        )
      }

    }
  }
}
