package javaapi


import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.util
import java.util.Arrays

import org.tensorflow._
import org.tensorflow.types.UInt8

object HopperTest extends App {
  LabelImage.apply()
}

object LabelImage {
  def apply(modelDir: String = "javaapi/resources", imageFile: String = "javaapi/resources/grace_hopper.jpg"): Unit = {
    val graphDef =
      Files.readAllBytes(Paths.get(modelDir, "tensorflow_inception_graph.pb"))
    val labels =
      Files.readAllLines(
        Paths.get(modelDir, "imagenet_comp_graph_label_strings.txt"))
    val imageBytes: Array[Byte] = Files.readAllBytes(Paths.get(imageFile))

    val image: Tensor[Float] = constructAndExecuteGraphToNormalizeImage(imageBytes)
    val labelProbabilities = executeInceptionGraph(graphDef, image)
    val bestLabelIdx = maxIndex(labelProbabilities)
    println(
      f"BEST MATCH: ${labels.get(bestLabelIdx)}%s (${labelProbabilities(bestLabelIdx) * 100f}%.2f%% likely)"
    )
  }

  def constructAndExecuteGraphToNormalizeImage(imageBytes: Array[Byte]): Tensor[Float] = {
    val g = new Graph()
    val b = new GraphBuilder(g)
    // Some constants specific to the pre-trained model at:
    // https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
    //
    // - The model was trained with images scaled to 224x224 pixels.
    // - The colors, represented as R, G, B in 1-byte each were converted to
    //   float using (value - Mean)/Scale.
    val H = 224
    val W = 224
    val mean: Float = 117f
    val scale: Float = 1f

    // Since the graph is being constructed once per execution here, we can use a constant for the
    // input image. If the graph were to be re-used for multiple input images, a placeholder would
    // have been more appropriate.
    val input: Output[String] =
      b.constant("input", imageBytes)
    val output: Output[Float] =
      b.div(
        b.sub(
          b.resizeBilinear(
            b.expandDims(b.cast[UInt8, Float](b.decodeJpeg(input, 3)),
                         b.constant("make_batch", 0)),
            b.constant("size", Array(H, W))),
          b.constant("mean", mean)
        ),
        b.constant("scale", scale)
      )

    val s = new Session(g)
    s.runner()
      .fetch(output.op().name())
      .run()
      .get(0)
      .asInstanceOf[Tensor[Float]] //.expect(Float.class);
  }

  def executeInceptionGraph(graphDef: Array[Byte],
                            image: Tensor[Float]): Array[Float] = {
    val g = new Graph()
    g.importGraphDef(graphDef)
    val s = new Session(g)
    val result: Tensor[Float] =
      s.runner()
        .feed("input", image)
        .fetch("output")
        .run()
        .get(0)
        .asInstanceOf[Tensor[Float]] //.expect(Float.class)
    val rshape: Array[Long] = result.shape()
    if (result.numDimensions() != 2 || rshape(0) != 1) {
      throw new RuntimeException(
        String.format(
          "Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
          util.Arrays.toString(rshape)))
    }
    val nlabels: Long = rshape(1)
    val arr: Array[Array[Float]] = Array.ofDim[Float](1, nlabels.toInt)
    result.copyTo(arr)(0)

  }

  def maxIndex(probabilities: Array[Float]): Int =
    probabilities.zipWithIndex.maxBy(_._1)._2
}

class GraphBuilder(val g: Graph) {
  def binaryOp[T](typ: String, in1: Output[T], in2: Output[T]): Output[T] =
    g.opBuilder(typ, typ).addInput(in1).addInput(in2).build().output[T](0)

  def binaryOp3[T, U, V](typ: String, in1: Output[U], in2: Output[V]): Output[T] =
    g.opBuilder(typ, typ).addInput(in1).addInput(in2).build().output[T](0)

  def constant[X, T](name: String, value: X)(implicit cls: ClassDataType[T],
                                             validity: ValidDataType[X, T]): Output[T] = {
    val t = Tensor.create(value)
    g.opBuilder("Const", name)
      .setAttr("dtype", ClassDataType.get[T])
      .setAttr("value", t)
      .build()
      .output[T](0)
  }

  def variable[T](name: String)(implicit cls: ClassDataType[T]): Output[T] = {
    g.opBuilder("Variable", name)
      .setAttr("dtype", ClassDataType.get[T])
      .setAttr("shape", Shape.make(1))
      .build()
      .output[T](0)
  }

  def cast[T, U: ClassDataType](value: Output[T]): Output[U] = {
    val dtype = ClassDataType.get[U]
    g.opBuilder("Cast", "Cast")
      .addInput(value)
      .setAttr("DstT", dtype)
      .build()
      .output[U](0)
  }

  def div(x: Output[Float], y: Output[Float]): Output[Float] =
    binaryOp("Div", x, y)

  def sub[T](x: Output[T], y: Output[T]): Output[T] =
    binaryOp("Sub", x, y)

  def expandDims[T](input: Output[T], dim: Output[Int]): Output[T] =
    binaryOp3("ExpandDims", input, dim)

  def resizeBilinear[T](images: Output[T], dim: Output[Int]): Output[Float] =
    binaryOp3("ResizeBilinear", images, dim)

  def decodeJpeg(contents: Output[String], channels: Long): Output[UInt8] =
    g.opBuilder("DecodeJpeg", "DecodeJpeg")
      .addInput(contents)
      .setAttr("channels", channels)
      .build()
      .output[UInt8](0)
}

trait ValidDataType[X, T] {
  val dt: DataType

  val cdt: ClassDataType[T]
}

object ValidDataType {
  def apply[X, T: ClassDataType](): ValidDataType[X, T] =
    new ValidDataType[X, T] {
      val cdt: ClassDataType[T] = implicitly[ClassDataType[T]]

      val dt: DataType = cdt.dt
    }

  implicit val int: ClassDataType[Int] = ClassDataType(DataType.INT32)
  implicit val long: ClassDataType[Long] = ClassDataType(DataType.INT64)
  implicit val float: ClassDataType[Float] = ClassDataType(DataType.FLOAT)
  implicit val double: ClassDataType[Double] = ClassDataType(DataType.DOUBLE)
  implicit val bool: ClassDataType[Boolean] = ClassDataType(DataType.BOOL)
  implicit val byte: ClassDataType[Byte] = ClassDataType(DataType.STRING)
  implicit val string: ClassDataType[String] = ClassDataType(DataType.STRING)
  implicit val arrayByte: ValidDataType[Array[Byte], String] = ValidDataType()
  implicit def array[T: ClassDataType]: ValidDataType[Array[T], T] =
    ValidDataType()
  // implicit def array[T](
  //     implicit base: ClassDataType[T]): ClassDataType[Array[T]] =
  //   ClassDataType(base.dt)

}

case class ClassDataType[T](dt: DataType) extends ValidDataType[T, T] {
  val cdt: ClassDataType[T] = this
}

object ClassDataType {
  def get[T: ClassDataType]: DataType = implicitly[ClassDataType[T]].dt
}
