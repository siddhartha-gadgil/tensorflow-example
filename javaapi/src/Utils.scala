package javaapi

import scala.jdk.CollectionConverters._
import scala.util.Using

import org.tensorflow._
import org.tensorflow.framework.optimizers.Optimizer
import org.tensorflow.ndarray._
import org.tensorflow.op._
import org.tensorflow.op.core.Placeholder
import org.tensorflow.op.math.Add
import org.tensorflow.types._

import ndarray._
import types._, family.TType
import core.Variable
import family.TType
import Optimizer.GradAndVar

object Utils {
  def opLookup(v: Operand[TFloat32], sess: Session): Float = {
    val result = sess.runner().fetch(v).run()
    val data = result.get(0).asInstanceOf[TFloat32]
    data.getFloat()
  }

  def dataLookup(v: Operand[TFloat32], sess: Session): TFloat32 = {
    val result = sess.runner().fetch(v).run()
    val data = result.get(0).asInstanceOf[TFloat32]
    data
  }

  def namedLookup(name: String, sess: Session): Float = {
    val result = sess.runner().fetch(name).run()
    val data = result.get(0).asInstanceOf[TFloat32]
    data.getFloat()
  }

  def minimizer[T <: TType](
      graph: Graph,
      optimizer: Optimizer,
      loss: Operand[T],
      variables: Array[Variable[T]],
      name: String
  ): Op = {
    val grads = graph
      .addGradients(loss.asOutput(), variables.map(x => x.asOutput()))
      .asInstanceOf[Array[Output[T]]]
    val gradsAndVars: List[GradAndVar[_ <: TType]] =
      grads.zip(variables).toList.map { case (g, v) =>
        new GradAndVar[T](g, v.asOutput())
      }
    optimizer.applyGradients(gradsAndVars.asJava, name)
  }
}
