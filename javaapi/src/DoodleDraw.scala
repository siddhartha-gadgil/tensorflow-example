package javaapi

import doodle.image._
import doodle.java2d._
// Colors and other useful stuff
import scala.collection.immutable.Nil
import doodle.core.Color

object DoodleDraw {
  import doodle.image.syntax._
  import doodle.java2d._
  import doodle.syntax._, doodle.image._
  import doodle.core._

  val frame = Frame.size(300, 100)

  lazy val canvas1: Canvas = effect.Java2dRenderer.canvas(frame).unsafeRunSync()

  // lazy val canvas2: Canvas = effect.Java2dRenderer.canvas(frame).unsafeRunSync

  // lazy val canvas3: Canvas = effect.Java2dRenderer.canvas(frame).unsafeRunSync

  def xyImage(
      xy: List[(Float, Float)],
      colour: Color = Color.hsl(0.degrees, 0.8, 0.6),
      colourSpin: Angle = Angle(0.2)
  ): Image =
    xy match {
      case Nil => Image.empty
      case head :: next =>
        Image
          .circle(5)
          .fillColor(colour)
          .at(Point(head._1, head._2))
          .on(xyImage(next, colour.spin(colourSpin), colourSpin))
    }

  def xyPlot(xy: List[(Float, Float)]) = xyImage(xy).draw()

  def stepsText(n: Int) =
    text[Algebra, Drawing](s"Steps: $n").font(
      font.Font.defaultSansSerif.size(30)
    )

  def showSteps(n: Int, canvas: Canvas = canvas1) =
    stepsText(n).drawWithCanvas(canvas)

  def linesImage(
      lines: List[((Float, Float), (Float, Float))],
      base: Image,
      colour: Color = Color.hsl(0.degrees, 0.8, 0.6),
      colourSpin: Angle = Angle(0.2)
  ): Image =
    lines match {
      case Nil => base
      case head :: next =>
        head match {
          case ((x1, y1), (x2, y2)) =>
            Image
              .line(x2 - x1, y2 - y1)
              .strokeColor(colour)
              .at(Point((x1 + x2) / 2, (y1 + y2) / 2))
              .on(linesImage(next, base, colour.spin(colourSpin)))
        }
    }

  def linesPlot(
      xy: List[(Float, Float)],
      lines: List[((Float, Float), (Float, Float))]
  ) =
    linesImage(lines, xyImage(xy)).draw()

}