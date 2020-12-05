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
  val windowWidth = 2

  val negSamples = 5

  val embedDim = 20

  val tokens: Vector[String] = os.read
    .lines(os.pwd / "javaapi" / "resources" / "shakespeare.txt")
    .flatMap { l =>
      l.split("\\s+").map(_.toLowerCase())
    }
    .toVector
  // .take(100)

  val paddedTokens = "<pad>" +: tokens

  val vocab: Map[String, Int] = paddedTokens.distinct.zipWithIndex.toMap

  val inverseVocab: Map[Int, String] = vocab.map { case (t, n) => (n, t) }

  val rnd = new Random()

  def randomMatrix(rows: Int, columns: Int): Array[Array[Float]] = {
    Array.fill(rows)(Array.fill(columns)(rnd.nextGaussian().toFloat))
  }

  def randomIndex() = rnd.nextInt(vocab.size)

  def negList(n: Int) = (0 until (n)).toList.map(_ => randomIndex())

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

  val skipGramIndices = positiveSkipGrams.map { case (word, context) =>
    (vocab(word), vocab(context))
  }

  def dataGroup(): Iterator[(Int, Int, List[Int])] = {
    Iterator.from(skipGramIndices).map { case (word, context) =>
      (word, context, negList(negSamples))
    }
  }

  def trainingData(epochs: Int): Iterator[(Int, Int, List[Int])] =
    Iterator.range(0, epochs).flatMap(_ => dataGroup())

  def trainedTensors(
      epochs: Int = 1
  ): Try[(Tensor[TFloat32], Tensor[TFloat32])] = Using(new Graph) { graph =>
    val w2v = new Word2Vec(graph, embedDim, vocab.size)
    val data = trainingData(epochs)
    println(s"training with data size ${skipGramIndices.size * epochs}")
    w2v.fit(data)
  }.flatten

}
class Word2Vec(
    graph: Graph,
    embedDim: Int,
    vocabSize: Int,
    learningRate: Float = 0.1f
) {
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

  val predictions = tf.math.sigmoid(dots)

  val cost1 = tf.math.neg(tf.math.mul(labels, tf.math.log(predictions)))

  val cost2 = tf.math.mul(
    tf.math.sub(tf.constant(1f), labels),
    tf.math.log(tf.math.sub(tf.constant(1f), predictions))
  )

  // max(x, 0) - x * z + log(1 + exp(-abs(x)))
  val stableCost = tf.math.add(
    tf.math.sub(tf.math.maximum(dots, tf.constant(0f)), tf.math.mul(dots, labels)),
    tf.math.log(
      tf.math.add(tf.constant(1f), tf.math.exp(tf.math.neg(tf.math.abs(dots))))
    )
  )

  val loss =
    tf.withName("loss")
      .reduceSum(
        // tf.math.sub(cost1, cost2),
        stableCost,
        tf.constant(0)
      )

  val optimizer = new Adam(graph, learningRate)

  val minimize = optimizer.minimize(loss)

  def fit(
      data: Iterator[(Int, Int, List[Int])]
  ): Try[(Tensor[TFloat32], Tensor[TFloat32])] = Using(new Session(graph)) {
    var count = 0
    session =>
      session.run(tf.init())
      data.foreach { case (word, posContext, negContext) =>
        if (count % 2000 == 0) println(s"feeding datapoint: $count")
        count += 1
        val words =
          TInt32.tensorOf(
            StdArrays.ndCopyOf(Array.fill(1 + negContext.size)(word))
          )
        val contexts =
          TInt32.tensorOf(
            StdArrays.ndCopyOf((posContext :: negContext).toArray)
          )
        val labelList =
          TFloat32.tensorOf(
            StdArrays.ndCopyOf(
              (1.0f :: List.fill(negContext.size)(0.0f)).toArray
            )
          )
        session
          .runner()
          .feed(wordIndices, words)
          .feed(contextIndices, contexts)
          .feed(labels, labelList)
          .addTarget(minimize)
          .run()
      }
      val output =
        session.runner().fetch(wordEmbed).fetch(contextEmbed).run()
      (
        output.get(0).expect(TFloat32.DTYPE),
        output.get(1).expect(TFloat32.DTYPE)
      )
  }

}

