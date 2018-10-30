import mill._, scalalib._

object javaapi extends ScalaModule{
  def scalaVersion = "2.12.6"
  def ivyDeps = Agg(
    ivy"org.tensorflow:tensorflow:1.11.0"
  )
}
