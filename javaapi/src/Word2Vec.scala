package javaapi

import org.tensorflow._
import org.tensorflow.op._
import org.tensorflow.types._
import scala.util.Using
import org.tensorflow.ndarray._
import org.tensorflow.framework.optimizers.Adam

import scala.util.{Try, Random}
import org.tensorflow.op.linalg.MatMul

object Word2Vec {
  val windowWidth = 2

  val tokens: Vector[String] = os.read
    .lines(os.pwd / "javaapi" / "resources" / "shakespeare.txt")
    .flatMap { l =>
      l.split("[\\s.!?\\-:,;]+").map(_.toLowerCase())
    }
    .toVector.map(_.replaceAll("""[\p{Punct}&&[^.]]""", ""))
  // .take(100)

  val vocabVector: Vector[String] = ("<pad>" +: tokens).distinct

  val vocab: Map[String, Int] = vocabVector.zipWithIndex.toMap

  val rnd = new Random()

  def randomMatrix(
      rows: Int,
      columns: Int,
      scale: Double
  ): Array[Array[Float]] = {
    Array.fill(rows)(Array.fill(columns)((rnd.nextGaussian() * scale).toFloat))
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

  def dataGroup(negSamples: Int): Iterator[(Int, Int, List[Int])] = {
    Iterator.from(skipGramIndices).map { case (word, context) =>
      (word, context, negList(negSamples))
    }
  }

  def trainingData(
      epochs: Int,
      negSamples: Int
  ): Iterator[(Int, Int, List[Int])] =
    Iterator.range(0, epochs).flatMap(_ => dataGroup(negSamples))

  def trainedTensors(
      epochs: Int = 1,
      embedDim: Int = 20,
      negSamples: Int = 2
  ): Try[(TFloat32, TFloat32)] = Using(new Graph) { graph =>
    val w2v = new Word2Vec(graph, embedDim, vocab.size)
    val data = trainingData(epochs, negSamples)
    println(s"Training with data size ${skipGramIndices.size * epochs}")
    w2v.fit(data)
  }.flatten

  def wordRepresentations(
      epochs: Int = 20,
      embedDim: Int = 128,
      negSamples: Int = 5
  ): WordRepresentations = {
    val (wordTensor, _) =
      trainedTensors(epochs, embedDim, negSamples).fold(
        { err =>
          println(err.getMessage())
          err.printStackTrace()
          throw err
        },
        identity(_)
      )
    val matrix = (0 until vocabVector.size).toVector.map { j =>
      val wordVec =
        (0 until embedDim).toVector.map(i => wordTensor.getFloat(i, j))
      wordVec
    }
    WordRepresentations(vocabVector, matrix, embedDim)
  }
}

class Word2Vec(
    graph: Graph,
    embedDim: Int,
    vocabSize: Int,
    learningRate: Float = 0.1f
) {
  val tf = Ops.create(graph)

  val wordIndices = tf.withName("word").placeholder(classOf[TInt32])

  val wordInputVec = tf.oneHot(
    wordIndices,
    tf.constant(vocabSize),
    tf.constant(1.0f),
    tf.constant(0.0f)
  )

  val contextIndices = tf.withName("context").placeholder(classOf[TInt32])

  val labels = // tf.constant(Array(0.0f, 1.0f))
    tf.withName("label").placeholder(classOf[TFloat32])

  val contextInputVec = tf.oneHot(
    contextIndices,
    tf.constant(vocabSize),
    tf.constant(1.0f),
    tf.constant(0.0f)
  )

  val wordEmbed =
    tf.variable(
      tf.constant(
        Word2Vec.randomMatrix(
          embedDim,
          vocabSize,
          scala.math.sqrt(1.0 / vocabSize)
        )
      )
    )

  val contextEmbed =
    tf.variable(
      tf.constant(
        Word2Vec.randomMatrix(
          embedDim,
          vocabSize,
          scala.math.sqrt(1.0 / vocabSize)
        )
      )
    )

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
    tf.math
      .sub(tf.math.maximum(dots, tf.constant(0f)), tf.math.mul(dots, labels)),
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
  ): Try[(TFloat32, TFloat32)] = Using(new Session(graph)) {
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
        output.get(0).asInstanceOf[TFloat32],
        output.get(1).asInstanceOf[TFloat32]
      )
  }

}

object WordRepresentations {
  def squaredDistance(v1: Vector[Float], v2: Vector[Float]): Float =
    v1.zip(v2).map { case (x, y) => (x - y) * (x - y) }.sum

  def pack(wr: WordRepresentations): (String, String) = (
    wr.matrix.map(_.mkString("\t")).mkString("\n"),
    wr.vocabVector.mkString("\n")
  )

  def save(name: String, wr: WordRepresentations): Unit = {
    val (mt, v) = pack(wr)
    os.write(os.pwd / "data" / name / "matrix.tsv", mt, createFolders = true)
    os.write(os.pwd / "data" / name / "vocabulary.txt", v, createFolders = true)
    os.write(
      os.pwd / "data" / name / "dim.txt",
      wr.dim.toString(),
      createFolders = true
    )
  }

  def unpack(mat: String, voc: String, dim: Int): WordRepresentations =
    WordRepresentations(
      voc.split("\n").toVector,
      mat.split("\n").toVector.map(s => s.split("\n").toVector.map(_.toFloat)),
      dim
    )

  def load(name: String): WordRepresentations = {
    unpack(
      os.read(os.pwd / "data" / name / "matrix.tsv"),
      os.read(os.pwd / "data" / name / "vocabulary.txt"),
      os.read(os.pwd / "data" / name / "dim.txt").toInt
    )
  }

  def oppositeVertex(
      v1: Vector[Float],
      v2: Vector[Float],
      v3: Vector[Float]
  ): Vector[Float] =
    v1.zip(v2.zip(v3)).map { case (x, (y, z)) => y + z - x }
}
case class WordRepresentations(
    vocabVector: Vector[String],
    matrix: Vector[Vector[Float]],
    dim: Int
) {
  import WordRepresentations._

  val vocab: Map[String, Int] = vocabVector.zipWithIndex.toMap

  val vec: Map[String, Vector[Float]] = vocabVector.zipWithIndex.map {
    case (w, n) => w -> matrix(n)
  }.toMap

  def nearestWord(v: Vector[Float]): String =
    vocabVector.minBy(w => squaredDistance(vec(w), v))

  def sortedWords(v: Vector[Float]): Vector[String] =
    vocabVector.sortBy(w => squaredDistance(vec(w), v))

}
