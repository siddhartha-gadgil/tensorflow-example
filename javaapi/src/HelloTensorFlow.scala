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

object HelloTensorFlow {
  def main(args: Array[String]) : Unit = {
    println("Hello TensorFlow " + TensorFlow.version())
    val x = TInt32.scalarOf(10)
    Using(ConcreteFunction.create(dblFunc)){dbl =>
    val dblX = dbl.call(x).expect(TInt32.DTYPE)
    println(x.data().getInt() + " doubled is " + dblX.data().getInt())
    }
  }

  val clTest = implicitly[Using.Releasable[java.lang.AutoCloseable]]

  def dblFunc(tf: Ops): Signature = {
    val x = tf.placeholder(TInt32.DTYPE);
    val dblX = tf.math.add(x, x);
    Signature.builder().input("x", x).output("dbl", dblX).build();
  }
}
