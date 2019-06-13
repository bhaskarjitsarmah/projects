package example

object Lesson1 {
  def sqrtIter(guess:Double, x:Double): Double =
    if(isGoodEnough(guess, x)) guess
    else sqrtIter(improve(guess, x), x)
}
