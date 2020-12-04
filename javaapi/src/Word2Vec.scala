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
import org.tensorflow.op.linalg.MatMul

object Word2Vec {
  val tokens: Vector[String] = os.read
    .lines(os.pwd / "javaapi" / "resources" / "shakespeare.txt")
    .flatMap { l =>
      l.split("\\s+")
    }
    .toVector
    .take(100)

  val paddedTokens = "<pad>" +: tokens

  val vocab: Map[String, Int] = paddedTokens.distinct.zipWithIndex.toMap

  val inverseVocab: Map[Int, String] = vocab.map { case (t, n) => (n, t) }

  val rnd = new Random()

  def randomMatrix(rows: Int, columns: Int): Array[Array[Float]] = {
    (0 until (rows)).map { i =>
      (0 until (columns)).map(_ => rnd.nextGaussian().toFloat).toArray
    }.toArray
  }

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

  lazy val oneHot = {
    Using(new Graph) { graph =>
      val w2v = new Word2VecExperiments(graph)
      Using(new Session(graph)) { session =>
        val output = session
          .runner()
          .feed("k", TInt32.tensorOf(StdArrays.ndCopyOf(Array(3, 2))))
          .fetch("onehot")
          .run()
        output.get(0).expect(TFloat32.DTYPE)
      }
    }
  }

  lazy val wTry = {
    Using(new Graph) { graph =>
      val w2v = new Word2VecExperiments(graph)
      Using(new Session(graph)) { session =>
        val output = session
          .runner()
          .feed("k", TInt32.scalarOf(3))
          .fetch("w")
          .fetch("sum")
          .run()
        (
          output.get(0).expect(TFloat32.DTYPE),
          output.get(1).expect(TFloat32.DTYPE)
        )
      }
    }
  }

  lazy val lossTry = {
    Using.Manager { use =>
      val graph = use(new Graph())
      val w2v = new Word2Vec(graph, 3, vocab.size)
      val session = use(new Session(graph))
      session.runner().addTarget("init").run()
      val output = session
        .runner()
        .feed("word", TInt32.tensorOf(StdArrays.ndCopyOf(Array(2, 4))))
        .feed("context", TInt32.tensorOf(StdArrays.ndCopyOf(Array(1, 5))))
        .feed("label", TFloat32.tensorOf(StdArrays.ndCopyOf(Array(0.0f, 1.0f))))
        .addTarget("init")
        .fetch("loss")
        .run()
      output.get(0).expect(TFloat32.DTYPE)

    }
  }

}

class Word2VecExperiments(graph: Graph) {
  val tf = Ops.create(graph)
  val k = tf.withName("k").placeholder(TInt32.DTYPE)
  val oh = tf
    .withName("onehot")
    .oneHot(k, tf.constant(7), tf.constant(1.0f), tf.constant(0f))

  val mat = tf.constant(
    Array(Array(1.0f, 2, 3, 4, 5, 6, 7), Array(8.0f, 9, 10, 11, 12, 13, 14))
  )

  val w = tf
    .withName("w")
    .reshape(
      tf.linalg.matMul(mat, tf.reshape(oh, tf.constant(Array(7, 1)))),
      tf.constant(Array(2))
    )

  val sum = tf
    .withName("sum")
    .reshape(
      tf.reduceSum(w, tf.constant(Array(0))),
      tf.constant(Array.emptyIntArray)
    )
}

class Word2Vec(graph: Graph, embedDim: Int, vocabSize: Int) {
  val tf = Ops.create(graph)

  val wordIndices = tf.withName("word").placeholder(TInt32.DTYPE)

  val wordInputVec = tf.oneHot(
    wordIndices,
    tf.constant(vocabSize),
    tf.constant(1.0f),
    tf.constant(0.0f)
  )

  val contextIndices = tf.withName("context").placeholder(TInt32.DTYPE)

  val labels = // tf.constant(Array(0.0f, 1.0f))
    tf.withName("label").placeholder(TFloat32.DTYPE)

  val contextInputVec = tf.oneHot(
    contextIndices,
    tf.constant(vocabSize),
    tf.constant(1.0f),
    tf.constant(0.0f)
  )

  val wordEmbed =
    tf.variable(tf.constant(Word2Vec.randomMatrix(embedDim, vocabSize)))

  val contextEmbed =
    tf.variable(tf.constant(Word2Vec.randomMatrix(embedDim, vocabSize)))

  val initialize = tf.withName("init").init()

  val wordVec = {
    tf
      .withName("wordvec")
      .linalg
      .matMul(
        wordEmbed,
        wordInputVec,
        MatMul.transposeA(false).transposeB(true)
      )
  }

  val contextVec = {
    tf.withName("contextvec")
      .linalg
      .matMul(
        contextEmbed,
        contextInputVec,
        MatMul.transposeA(false).transposeB(true)
      )
  }

  val dots = tf
    .withName("dots")
    .reduceSum(tf.math.mul(wordVec, contextVec), tf.constant(0))

  // val ytrue = tf.withName("ytrue").math.add(labels, tf.constant(Array(0.0f, 0.0f)))

  val predictions = tf.math.sigmoid(dots)

  val cost1 = tf.math.neg(tf.math.mul(labels, tf.math.log(predictions)))

  val cost2 = tf.math.mul(
    tf.math.sub(tf.onesLike(labels), labels),
    tf.math.log(tf.math.sub(tf.onesLike(predictions), predictions))
  )

  val losses =
     tf.withName("loss").reduceSum(tf.math.sub(cost1, cost2), tf.constant(0))

}
