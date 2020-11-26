package javaapi

import org.tensorflow.ConcreteFunction
import org.tensorflow.Signature
import org.tensorflow.Tensor
import org.tensorflow.TensorFlow
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Placeholder
import org.tensorflow.op.math.Add
import org.tensorflow.types.TInt32
import scala.util.Using
import org.tensorflow.{Session, Graph, Tensor, _}
import org.tensorflow.op.linalg.MatMul
import org.tensorflow.ndarray._

object HelloTensorFlow {
  def main(args: Array[String]) : Unit = {
    run()
    // GradientDescentExample.run()
    MSEGradientDescent.run()
    // GeometricSimple.run()
    SimpleLinearModel.run()
    LabelImage.run()
  }

  def run() = {
    println("Hello TensorFlow " + TensorFlow.version())
    val x = TInt32.scalarOf(10)
    Using(ConcreteFunction.create(dblFunc)) { dbl =>
      val dblX = dbl.call(x).expect(TInt32.DTYPE)
      println(x.data().getInt() + " doubled is " + dblX.data().getInt())
    }
    println("Session should output 31")
    Using.Manager { use =>
      val g = use(new Graph())
      val s = use(new Session(g))
      val tf = Ops.create(g)
      transpose_A_times_X(tf, Array(Array(2), Array(3)))
      Using(TInt32.tensorOf(StdArrays.ndCopyOf(Array(Array(5), Array(7))))) {
        x =>
          val outputs = s.runner().feed("X", x).fetch("Y").run()
          println(outputs.get(0).expect(TInt32.DTYPE).data().getInt(0, 0))
      }
    }

  }

  val clTest = implicitly[Using.Releasable[java.lang.AutoCloseable]]

  def dblFunc(tf: Ops): Signature = {
    val x = tf.placeholder(TInt32.DTYPE);
    val dblX = tf.math.add(x, x);
    Signature.builder().input("x", x).output("dbl", dblX).build();
  }

  def transpose_A_times_X(tf: Ops, a: Array[Array[Int]]) {
    tf.withName("Y")
      .linalg
      .matMul(
        tf.withName("A").constant(a),
        tf.withName("X").placeholder(TInt32.DTYPE),
        MatMul.transposeA(true).transposeB(false)
      );
  }
}
