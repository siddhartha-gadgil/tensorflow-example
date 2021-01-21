import mill._, scalalib._

val scalaV = "2.13.3"
object javaapi extends ScalaModule{
  def scalaVersion = scalaV
  def ivyDeps = Agg(
    ivy"org.tensorflow:tensorflow-core-platform:0.2.0",
    ivy"org.tensorflow:tensorflow-framework:0.2.0",
    ivy"com.lihaoyi::os-lib:0.7.1",
    ivy"org.creativescala::doodle:0.9.21"
  )
}

object oldjavaapi extends ScalaModule{
  def scalaVersion = scalaV
  def ivyDeps = Agg(
    ivy"org.tensorflow:tensorflow:1.11.0"
  )
}


object scalaapi extends ScalaModule{
  def scalaVersion = "2.12.10"
  def ivyDeps = Agg(
    ivy"org.platanios::tensorflow:0.4.0;classifier=linux-cpu-x86_64"
  )
}
