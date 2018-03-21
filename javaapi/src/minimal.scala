object Minimal extends App {
  import org.tensorflow._
  val value = "Hello from " + TensorFlow.version();
  val t = Tensor.create(value.getBytes("UTF-8"))
  val g = new Graph
  g.opBuilder("Const", "MyConst").setAttr("dtype", t.dataType()).setAttr("value", t).build();
  val s = new Session(g)
  val output = s.runner().fetch("MyConst").run().get(0)
  println(new String(output.bytesValue(), "UTF-8"))
}
