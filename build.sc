import mill._, scalalib._

object javaapi extends ScalaModule{
  def scalaVersion = "2.12.6"
  def ivyDeps = Agg(
    ivy"org.tensorflow:tensorflow:1.11.0"
  )
}

object scalaapi extends ScalaModule{
  def scalaVersion = "2.12.6"
  def ivyDeps = Agg(
    ivy"org.platanios::tensorflow:0.4.0;classifier=linux-cpu-x86_64"
  )
}
