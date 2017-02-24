import BIDMat.SciFunctions._
import YADLL.OperatorGraph.{BuildArch, Theta}
import YAVL.TextStream.Dict.Lexicon

import scala.runtime.RichInt
//Imports from BIDMat
import BIDMat.{CMat, CSMat, DMat, Dict, FMat, GIMat, GMat, GSMat, HMat, IDict, IMat, Mat, SBMat, SDMat, SMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._

/**
  * Created by Alex on 2/23/2017.
  */
object LatentVarProjector {
  Mat.checkMKL //<-- needed in order to check for BIDMat on cpu or gpu...

  def main(args : Array[String]): Unit = {
    if (args.length < 3) {
      System.err.println("usage: [k] [/path/to/out-name.mat] [/path/to/matrix.mat?theta.theta] ?[paramName?]")
      return
    }
    val k = args(0).toInt
    var samps : Mat = null
    if(args.length == 4){
      val archBuilder = new BuildArch()
      val theta = archBuilder.loadTheta(args(2))
      val paramName = args(3)
      samps = theta.getParam(paramName).t
      println(" > Loaded from Theta, matrix of size = "+size(samps))
      //println(theta.printDims())
    }else{
      samps = HMat.loadFMat(args(2)) //<-- get column-major matrix of latent var vectors
      println(" > Loaded matrix of size = "+size(samps))
    }

    //Perform rotation after finding eigen-vectors of the data
    //val orig = samps.copy
    val A = samps * samps.t
    val stat = seig(A,true)
    val eigens = stat._2.asInstanceOf[Mat] //get eigen-vectors
    val components = eigens(?,new RichInt(0) until k)
    samps = components.t * samps //rotate decoder
    println("comps.size = "+size(components.t))
    println("X-Rot.size = "+size(samps))

    println(" > Writing to disk: "+args(1))
    HMat.saveFMatTxt(args(1),FMat(samps.t))
    println(" > Out.shape = "+size(samps.t))

    /* PCA checks for correctness
    println((eigens * eigens.t))
    println((eigens * eigens.t) * orig)
    println(orig)
    */
  }
}
