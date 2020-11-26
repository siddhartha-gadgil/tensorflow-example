package javaapi

import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.util
import java.util.Arrays

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
import org.tensorflow.op.image.DecodeJpeg

object LabelImage {
  def maxIndex(probabilities: Array[Float]): Int =
    probabilities.zipWithIndex.maxBy(_._1)._2

  val modelDir = "javaapi/resources"
  val labels =
    Files.readAllLines(
      Paths.get(modelDir, "imagenet_comp_graph_label_strings.txt")
    )

  val imageFile: String = "javaapi/resources/grace_hopper.jpg"

  def constructAndExecuteGraphToNormalizeImage(
      filename: String
  ): Tensor[TFloat32] = {
    val graph = new Graph()
    val tf = Ops.create(graph)
    val imageInput = tf.io.readFile(tf.constant(imageFile))

    val H = 224
    val W = 224
    val mean: Float = 117f
    val scale: Float = 1f

    val image =
      tf.math.div(
        tf.math.sub(
          tf.image.resizeBilinear(
            tf.expandDims(
              (tf.image.decodeJpeg(
                imageInput,
                DecodeJpeg.channels(3)
              )),
              tf.constant(0)
            ),
            tf.constant(Array(H, W))
          ),
          tf.constant(mean)
        ),
        tf.constant(scale)
      )
    val session = new Session(graph)
    session.runner().fetch(image).run().get(0).expect(TFloat32.DTYPE)
  }

  val savedModelBundle =
    SavedModelBundle.load(modelDir, SavedModelBundle.DEFAULT_TAG)

  val modelSession = savedModelBundle.session()

  val image = constructAndExecuteGraphToNormalizeImage(imageFile)

  val labelProbabilityTensor = modelSession
    .runner()
    .feed("input", image)
    .fetch("output")
    .run()
    .get(0)
    .expect(TFloat32.DTYPE)
    .data

    def run() = {
        println(s"Tensor: $labelProbabilityTensor")
    }

}
