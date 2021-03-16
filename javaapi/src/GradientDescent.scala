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

object GradientDescentExample {
  val var0Init = Array(1.0f, 2.0f)
  val var1Init = Array(3.0f, 4.0f)
  val grads0Init = Array(0.1f, 0.1f)
  val grads1Init = Array(0.01f, 0.01f)

  val learningRate = 3.0f

  val shape0 = Shape.of(var0Init.length)
  val shape1 = Shape.of(var1Init.length)

  def run(): Unit = {
    Using.Manager { use =>
      val graph = use(new Graph())
      val session = use(new Session(graph))
      val tf = Ops.create(graph)

      val var0 = tf.withName("var0").variable(shape0, classOf[TFloat32])
      val var1 = tf.withName("var1").variable(shape1, classOf[TFloat32])

      val var0Initializer = tf.assign(var0, tf.constant(var0Init))
      val var1Initializer = tf.assign(var1, tf.constant(var1Init))

      val grads0 = tf.constant(grads0Init)
      val grads1 = tf.constant(grads1Init)

      val gradsAndVars: List[GradAndVar[_ <: TType]] = List(
        new Optimizer.GradAndVar(grads0.asOutput(), var0.asOutput()),
        new Optimizer.GradAndVar(grads1.asOutput(), var1.asOutput())
      )

      val instance = new GradientDescent(graph, learningRate)
      val update = instance.applyGradients(gradsAndVars.asJava, "SGDTest")

      /* initialize the local variables */
      session.run(var0Initializer);
      session.run(var1Initializer);

      session.run(update)
      session.run(update)

      /* initialize the accumulators */
      val result = session.runner().addTarget(tf.init()).fetch("var0").run();
      def res(n: Int) = result.get(0).asInstanceOf[TFloat32].getFloat(n)
      println(s"ran session, fetched var0 as ${res(0)} and ${res(1)}")

    }
  }
}
