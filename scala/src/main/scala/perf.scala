// This file was formerly a part of Julia. License is MIT: https://julialang.org/license

import java.io.{BufferedOutputStream, FileOutputStream, PrintStream}
import java.util.Random

import breeze.linalg._
import breeze.stats.distributions.Gaussian
import breeze.stats.distributions.Rand.VariableSeed.randBasis

object Perf {
  val NITER = 5

  // Volatile sink to prevent the JVM JIT from optimizing away benchmark computations.
  @volatile var sink: Any = _

  def printPerf(name: String, t: Long): Unit = {
    printf("scala,%s,%.6f\n", name, t / 1e6)
  }

  // ---------------------------------------------------------------------------
  // Fibonacci
  // ---------------------------------------------------------------------------

  def fib(n: Int): Int = if (n < 2) n else fib(n - 1) + fib(n - 2)

  // ---------------------------------------------------------------------------
  // Mandelbrot
  // ---------------------------------------------------------------------------

  def mandel(zr0: Double, zi0: Double): Int = {
    val maxiter = 80
    var zr = zr0
    var zi = zi0
    var n = 0
    while (n < maxiter) {
      if (zr * zr + zi * zi > 4.0) return n
      val zrNew = zr * zr - zi * zi + zr0
      val ziNew = 2.0 * zr * zi + zi0
      zr = zrNew
      zi = ziNew
      n += 1
    }
    maxiter
  }

  def mandelperf(): Array[Int] = {
    val M = new Array[Int](21 * 26)
    var i = 0
    while (i < 21) {
      var j = 0
      while (j < 26) {
        M(26 * i + j) = mandel((j - 20) / 10.0, (i - 10) / 10.0)
        j += 1
      }
      i += 1
    }
    M
  }

  // ---------------------------------------------------------------------------
  // Quicksort
  // ---------------------------------------------------------------------------

  def quicksort(a: Array[Double], lo0: Int, hi: Int): Unit = {
    var i = lo0
    var j = hi
    var lo = lo0
    while (i < hi) {
      val pivot = a((lo + hi) / 2)
      while (i <= j) {
        while (a(i) < pivot) i += 1
        while (a(j) > pivot) j -= 1
        if (i <= j) {
          val tmp = a(i)
          a(i) = a(j)
          a(j) = tmp
          i += 1
          j -= 1
        }
      }
      if (lo < j) quicksort(a, lo, j)
      lo = i
      j = hi
    }
  }

  // ---------------------------------------------------------------------------
  // Pi sum
  // ---------------------------------------------------------------------------

  def pisum(): Double = {
    var sum = 0.0
    var j = 0
    while (j < 500) {
      sum = 0.0
      sink = sum // prevent the JIT from collapsing the outer loop
      var k = 1
      while (k <= 10000) {
        sum += 1.0 / (k * k)
        k += 1
      }
      j += 1
    }
    sum
  }

  // ---------------------------------------------------------------------------
  // Print to file
  // ---------------------------------------------------------------------------

  def printfd(n: Int): Unit = {
    val ps = new PrintStream(new BufferedOutputStream(new FileOutputStream("/dev/null")))
    try {
      var i = 0
      while (i < n) {
        ps.println(s"$i ${i + 1}")
        i += 1
      }
    } finally {
      ps.close()
    }
  }

  // ---------------------------------------------------------------------------
  // Random matrix statistics (Breeze / BLAS)
  // ---------------------------------------------------------------------------

  def randmatstat(t: Int): (Double, Double) = {
    val n = 5
    val v = new Array[Double](t)
    val w = new Array[Double](t)
    val g = Gaussian(0, 1)

    for (i <- 0 until t) {
      val a = DenseMatrix.rand[Double](n, n, g)
      val b = DenseMatrix.rand[Double](n, n, g)
      val c = DenseMatrix.rand[Double](n, n, g)
      val d = DenseMatrix.rand[Double](n, n, g)

      val P = DenseMatrix.zeros[Double](n, 4 * n)
      P(::, 0 until n) := a
      P(::, n until 2 * n) := b
      P(::, 2 * n until 3 * n) := c
      P(::, 3 * n until 4 * n) := d

      val Q = DenseMatrix.zeros[Double](2 * n, 2 * n)
      Q(0 until n, 0 until n) := a
      Q(0 until n, n until 2 * n) := b
      Q(n until 2 * n, 0 until n) := c
      Q(n until 2 * n, n until 2 * n) := d

      val V = P.t * P
      val W = Q.t * Q

      v(i) = trace(V * V * V * V)
      w(i) = trace(W * W * W * W)
    }

    def stdev(arr: Array[Double]): Double = {
      val mean = arr.sum / arr.length
      val variance = arr.map(x => (x - mean) * (x - mean)).sum / (arr.length - 1)
      math.sqrt(variance)
    }

    def mean(arr: Array[Double]): Double = arr.sum / arr.length

    (stdev(v) / mean(v), stdev(w) / mean(w))
  }

  // ---------------------------------------------------------------------------
  // Random matrix multiply (Breeze / BLAS)
  // ---------------------------------------------------------------------------

  def randmatmul(n: Int): DenseMatrix[Double] = {
    val a = DenseMatrix.rand[Double](n, n)
    val b = DenseMatrix.rand[Double](n, n)
    a * b
  }

  // ---------------------------------------------------------------------------
  // Main
  // ---------------------------------------------------------------------------

  def main(args: Array[String]): Unit = {
    val rand = new Random(0)
    var tmin = 0L
    var t = 0L

    // recursion_fibonacci
    assert(fib(20) == 6765)
    tmin = Long.MaxValue
    for (_ <- 0 until NITER) {
      t = System.nanoTime()
      sink = fib(20)
      tmin = math.min(tmin, System.nanoTime() - t)
    }
    printPerf("recursion_fibonacci", tmin)

    // parse_integers
    tmin = Long.MaxValue
    for (_ <- 0 until NITER) {
      t = System.nanoTime()
      for (_ <- 0 until 1000) {
        val n = rand.nextInt(Integer.MAX_VALUE)
        val s = Integer.toHexString(n)
        val m = Integer.parseInt(s, 16)
        assert(m == n)
      }
      tmin = math.min(tmin, System.nanoTime() - t)
    }
    printPerf("parse_integers", tmin)

    // userfunc_mandelbrot
    {
      val M = mandelperf()
      val s = M.sum
      assert(s == 14791, s"mandel sum was $s, expected 14791")
    }
    tmin = Long.MaxValue
    for (_ <- 0 until NITER) {
      t = System.nanoTime()
      sink = mandelperf()
      tmin = math.min(tmin, System.nanoTime() - t)
    }
    printPerf("userfunc_mandelbrot", tmin)

    // recursion_quicksort
    tmin = Long.MaxValue
    for (_ <- 0 until NITER) {
      t = System.nanoTime()
      val d = new Array[Double](5000)
      var j = 5000
      while (j > 0) {
        j -= 1
        d(j) = rand.nextDouble()
      }
      quicksort(d, 0, 4999)
      sink = d
      tmin = math.min(tmin, System.nanoTime() - t)
    }
    printPerf("recursion_quicksort", tmin)

    // iteration_pi_sum
    tmin = Long.MaxValue
    for (_ <- 0 until NITER) {
      t = System.nanoTime()
      val pi = pisum()
      tmin = math.min(tmin, System.nanoTime() - t)
      assert(math.abs(pi - 1.644834071848065) < 1e-12)
    }
    printPerf("iteration_pi_sum", tmin)

    // matrix_statistics
    tmin = Long.MaxValue
    for (_ <- 0 until NITER) {
      t = System.nanoTime()
      sink = randmatstat(1000)
      tmin = math.min(tmin, System.nanoTime() - t)
    }
    printPerf("matrix_statistics", tmin)

    // matrix_multiply
    tmin = Long.MaxValue
    for (_ <- 0 until NITER) {
      t = System.nanoTime()
      val m = randmatmul(1000)
      assert(0 <= m(0, 0))
      tmin = math.min(tmin, System.nanoTime() - t)
    }
    printPerf("matrix_multiply", tmin)

    // print_to_file
    tmin = Long.MaxValue
    for (_ <- 0 until NITER) {
      t = System.nanoTime()
      printfd(100000)
      tmin = math.min(tmin, System.nanoTime() - t)
    }
    printPerf("print_to_file", tmin)

    // nbody
    val nbRef = nbodyPerf(1000, 10, 0.01)
    assert(math.abs(nbodyPerf(1000, 10, 0.01) - nbRef) < 1e-6)
    var nbCs = 0.0
    tmin = Long.MaxValue
    for (_ <- 0 until NITER) {
      t = System.nanoTime()
      nbCs += nbodyPerf(1000, 10, 0.01)
      tmin = math.min(tmin, System.nanoTime() - t)
    }
    sink = nbCs
    printPerf("simulation_nbody", tmin)
  }

  // N-body simulation
  def nbodyInit(n: Int): Array[Array[Double]] = { val r = new scala.util.Random(42); val invN = 1.0/n; val x=Array.fill(n)(r.nextDouble()*2-1);val y=Array.fill(n)(r.nextDouble()*2-1);val z=Array.fill(n)(r.nextDouble()*2-1);val vx=Array.fill(n)((r.nextDouble()-0.5)*0.1);val vy=Array.fill(n)((r.nextDouble()-0.5)*0.1);val vz=Array.fill(n)((r.nextDouble()-0.5)*0.1);val m=Array.fill(n)(r.nextDouble()*invN);Array(x,y,z,vx,vy,vz,m) }
  def nbodyStep(s: Array[Array[Double]], dt: Double): Unit = { val n=s(0).length;val G=1.0;val eps2=1e-4;val x=s(0);val y=s(1);val z=s(2);val vx=s(3);val vy=s(4);val vz=s(5);val m=s(6);for(i<-0 until n){var fx=0.0;var fy=0.0;var fz=0.0;val xi=x(i);val yi=y(i);val zi=z(i);for(j<-0 until n){val dx=x(j)-xi;val dy=y(j)-yi;val dz=z(j)-zi;val dsq=dx*dx+dy*dy+dz*dz+eps2;val inv=1.0/math.sqrt(dsq);val inv3=inv*inv*inv;fx+=dx*inv3*m(j);fy+=dy*inv3*m(j);fz+=dz*inv3*m(j)};vx(i)+=dt*G*fx;vy(i)+=dt*G*fy;vz(i)+=dt*G*fz};for(i<-0 until n){x(i)+=dt*vx(i);y(i)+=dt*vy(i);z(i)+=dt*vz(i)} }
  def nbodyPerf(n: Int, steps: Int, dt: Double): Double = { val s = nbodyInit(n); for (_ <- 0 until steps) nbodyStep(s, dt); var cs=0.0; for (i <- 0 until n) cs+=s(0)(i)+s(1)(i)+s(2)(i); cs }
}
