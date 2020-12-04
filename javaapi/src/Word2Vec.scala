package javaapi

import org.tensorflow._, ndarray._, types._
import org.tensorflow.op._, core.Variable
import org.tensorflow.op.core.Placeholder
import org.tensorflow.op.math.Add
import org.tensorflow.types._, family.TType
import scala.util.Using
import org.tensorflow.ndarray._
import org.tensorflow.framework.optimizers.{
  Optimizer,
  GradientDescent,
  AdaGrad,
  AdaDelta,
  Adam
}
import scala.jdk.CollectionConverters._
import org.tensorflow.framework.optimizers.AdaGrad

import Optimizer.GradAndVar
import scala.jdk.CollectionConverters._
import Utils._
import scala.util.{Try, Random}

object Word2Vec {
  val tokens: Vector[String] =  os.read
    .lines(os.pwd / "javaapi" / "resources" / "shakespeare.txt")
    .flatMap { l =>
      l.split("\\s+")
    }
    .toVector

  val paddedTokens = "<pad>" +: tokens

  val vocab: Map[String, Int] = paddedTokens.distinct.zipWithIndex.toMap

  val inverseVocab: Map[Int, String] = vocab.map { case (t, n) => (n, t) }

  val rnd = new Random()

  val windowWidth = 2

  def getOrderedSkipGrams(tokenVec: Vector[String]): Vector[(String, String)] =
    for {
      (first, n) <- tokenVec.zipWithIndex
      second <- tokenVec.drop(n + 1).take(windowWidth)
    } yield (first, second)

  def getSkipGrams(tokenVec: Vector[String]): Vector[(String, String)] =
    getOrderedSkipGrams(tokenVec).flatMap { case (first, second) =>
      Vector(first -> second, second -> first)
    }

  val positiveSkipGrams = getSkipGrams(tokens)

}
