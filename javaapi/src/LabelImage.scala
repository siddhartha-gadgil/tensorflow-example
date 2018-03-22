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

  def constructAndExecuteGraphToNormalizeImage(imageBytes : Array[Byte])  = {
      val g = new Graph()
      val b = new GraphBuilder(g);
      // Some constants specific to the pre-trained model at:
      // https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
      //
      // - The model was trained with images scaled to 224x224 pixels.
      // - The colors, represented as R, G, B in 1-byte each were converted to
      //   float using (value - Mean)/Scale.
      val H = 224;
      val W = 224;
      val mean : Float = 117f;
      val scale : Float = 1f;

      // Since the graph is being constructed once per execution here, we can use a constant for the
      // input image. If the graph were to be re-used for multiple input images, a placeholder would
      // have been more appropriate.
      val input : Output[String] = b.constant[Array[Byte], String]("input", imageBytes);
      val output : Output[Float] =
          b.div(
              b.sub(
                  b.resizeBilinear(
                      b.expandDims(
                          b.cast[UInt8, Float](b.decodeJpeg(input, 3)),
                          b.constant[Int, Int]("make_batch", 0)),
                      b.constant[Array[Int], Int]("size", Array(H, W))),
                  b.constant[Float, Float]("mean", mean)),
              b.constant[Float, Float]("scale", scale))

      val  s = new Session(g)
      s.runner().fetch(output.op().name()).run().get(0)//.expect(Float.class);
  }

  def maxIndex(probabilities: Array[Float]) =
    probabilities.zipWithIndex.maxBy(_._1)._2
}

class GraphBuilder(val g: Graph) {
  def binaryOp[T](typ: String, in1: Output[T], in2: Output[T]) =
    g.opBuilder(typ, typ).addInput(in1).addInput(in2).build().output[T](0)

  def binaryOp3[T, U, V](typ: String, in1: Output[U], in2: Output[V]) =
    g.opBuilder(typ, typ).addInput(in1).addInput(in2).build().output[T](0)


  // FIXME: should infer `T` from `X`
  def constant[X, T](name: String, value: X)(implicit cls: ClassDataType[T], validity: ValidDataType[T, T]) = {
    val t = Tensor.create(value)
    g.opBuilder("Const", name)
      .setAttr("dtype", ClassDataType.get[T])
      .setAttr("value", t)
      .build()
      .output[T](0)
  }

  def cast[T, U: ClassDataType](value: Output[T]) = {
    val dtype = ClassDataType.get[U]
    g.opBuilder("Cast", "Cast")
      .addInput(value)
      .setAttr("DstT", dtype)
      .build()
      .output[U](0)
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

trait ValidDataType[T, X]{
  val dt : DataType
}

object ValidDataType{
  implicit val int: ClassDataType[Int] = ClassDataType(DataType.INT32)
  implicit val long: ClassDataType[Long] = ClassDataType(DataType.INT64)
  implicit val float: ClassDataType[Float] = ClassDataType(DataType.FLOAT)
  implicit val double: ClassDataType[Double] = ClassDataType(DataType.DOUBLE)
  implicit val bool: ClassDataType[Boolean] = ClassDataType(DataType.BOOL)
  implicit val byte: ClassDataType[Byte] = ClassDataType(DataType.STRING)
  implicit val string: ClassDataType[String] = ClassDataType(DataType.STRING)
  implicit val arrayByte: ClassDataType[Array[Byte]] = ClassDataType(DataType.STRING)
  implicit def array[T](
      implicit base: ClassDataType[T]): ClassDataType[Array[T]] =
    ClassDataType(base.dt)

}

case class ClassDataType[T](dt: DataType) extends ValidDataType[T, T]

object ClassDataType{
  def get[T: ClassDataType] = implicitly[ClassDataType[T]].dt
}
