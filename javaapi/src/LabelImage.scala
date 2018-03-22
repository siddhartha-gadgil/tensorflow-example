package javaapi

import java.io.IOException
import java.io.PrintStream
import java.nio.charset.Charset
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.util.Arrays
import java.util.List
import org.tensorflow.DataType
import org.tensorflow.Graph
import org.tensorflow.Output
import org.tensorflow.Session
import org.tensorflow.Tensor
import org.tensorflow.TensorFlow
import org.tensorflow.types.UInt8

object LabelImage {
  def apply(modelDir: String, imageFile: String): Unit = {
    val graphDef =
      Files.readAllBytes(Paths.get(modelDir, "tensorflow_inception_graph.pb"))
    val labels =
      Files.readAllLines(
        Paths.get(modelDir, "imagenet_comp_graph_label_strings.txt"))
    val imageBytes = Files.readAllBytes(Paths.get(imageFile))
  }

  def maxIndex(probabilities: Array[Float]) =
    probabilities.zipWithIndex.maxBy(_._1)._2
}

class GraphBuilder(val g: Graph) {
  def binaryOp[T](typ: String, in1: Output[T], in2: Output[T]) =
    g.opBuilder(typ, typ).addInput(in1).addInput(in2).build().output[T](0)

  def binaryOp3[T, U, V](typ: String, in1: Output[U], in2: Output[V]) =
    g.opBuilder(typ, typ).addInput(in1).addInput(in2).build().output[T](0)

  def constant[T: ClassDataType](name: String, value: T) = {
    val t = Tensor.create(value)
    g.opBuilder("Const", name)
      .setAttr("dtype", ClassDataType.get[T])
      .setAttr("value", t)
      .build()
      .output[T](0)
  }

  def cast[T, U: ClassDataType](value: Output[T]) {
    val dtype = ClassDataType.get[U]
    g.opBuilder("Cast", "Cast")
      .addInput(value)
      .setAttr("DstT", dtype)
      .build()
      .output[U](0);
  }

  def div(x: Output[Float], y: Output[Float]) =
    binaryOp("Div", x, y)

  def sub[T](x: Output[T], y: Output[T]) =
    binaryOp("Sub", x, y)

  def expandDims[T](input: Output[T], dim: Output[Int]): Output[T] =
    binaryOp3("ExpandDims", input, dim)

  def resizeBilinear[T](images: Output[T], dim: Output[Int]): Output[Float] =
    binaryOp3("ResizeBilinear", images, dim)

  def decodeJpeg(contents: Output[String], channels: Long) =
    g.opBuilder("DecodeJpeg", "DecodeJpeg")
        .addInput(contents)
        .setAttr("channels", channels)
        .build()
        .output[UInt8](0)
}

case class ClassDataType[T](val dt: DataType)

object ClassDataType {
  def get[T: ClassDataType] = implicitly[ClassDataType[T]].dt

  implicit val int: ClassDataType[Int] = ClassDataType(DataType.INT32)
  implicit val long: ClassDataType[Long] = ClassDataType(DataType.INT64)
  implicit val float: ClassDataType[Float] = ClassDataType(DataType.FLOAT)
  implicit val double: ClassDataType[Double] = ClassDataType(DataType.DOUBLE)
  implicit val bool: ClassDataType[Boolean] = ClassDataType(DataType.BOOL)
  implicit val byte: ClassDataType[Byte] = ClassDataType(DataType.STRING)
  implicit def array[T](
      implicit base: ClassDataType[T]): ClassDataType[Array[T]] =
    ClassDataType(base.dt)

}
