import mill._, scalalib._

object javaapi extends ScalaModule{
  def scalaVersion = "2.12.6"
  def ivyDeps = Agg(
    ivy"org.tensorflow:tensorflow:1.8.0"
  )
}

import mill.eval._, mill.define._

def ensimeConfig(ev: Evaluator[Any]) = T.command{
  val root = ev.rootModule
  val modules =
    ev.rootModule.millInternal.segmentsToModules
      .values.collect{ case x: scalalib.ScalaModule => (x.millModuleSegments, x)}
      .toSeq

  val resolved = for((path, mod) <- modules) yield {
      val allIvyDeps = T.task{mod.transitiveIvyDeps() ++ mod.scalaLibraryIvyDeps() ++ mod.compileIvyDeps()}
      val externalDependencies = T.task{
        mod.resolveDeps(allIvyDeps)() ++
        Task.traverse(mod.transitiveModuleDeps)(_.unmanagedClasspath)().flatten
      }
      externalDependencies
    }
  resolved(0).map{(x) => println("external dependency")}

  println("blah")
  println(modules)
}
