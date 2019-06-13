object session {

  def sqrtIter(guess:Double, x:Double): Double =
    if(isGoodEnough(guess, x)) guess
    else sqrtIter(improve(guess, x), x)

  def isGoodEnough(guess:Double, x:Double) =
    (guess * guess - x) < 0.001

  def improve(guess:Double, x:Double) =
    (guess + x/guess) / 2

  sqrtIter(1, 2)

}