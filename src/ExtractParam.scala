//General imports
import java.io.File
import java.util.HashMap
import java.util.ArrayList
import java.util.Random

import scala.io.StdIn.{readInt, readLine}
import BIDMat.FMat
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import YADLL.OperatorGraph.Graph.OGraph
import YADLL.OperatorGraph.Operators.SpecOps.{KL_Gauss, KL_Piece}
import YADLL.OperatorGraph.Optimizers.{SGOpt, _}
import YADLL.OperatorGraph.Theta
import YADLL.OperatorGraph.Operators.SpecOps.KL_Piece
import YADLL.Utils.{MathUtils, MiscUtils}
import YAVL.TextStream.Dict.Lexicon
import YAVL.Utils.{Logger, ScalaDebugUtils}

import scala.runtime.{RichInt, RichInt$}
//Imports from BIDMat
import BIDMat.{CMat, CSMat, DMat, Dict, FMat, GIMat, GMat, GSMat, HMat, IDict, IMat, Mat, SBMat, SDMat, SMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
//Imports from the library YADLL v0.7
import YADLL.OperatorGraph.BuildArch
import YADLL.Utils.ConfigFile

/**
  * Created by ago109 on 2/24/17.
  */
object ExtractParam {

  def main(args : Array[String]): Unit ={
    if(args.length != 4){
      System.err.println("usage: [theta] [paramName] [outFname] [tranpose?]")
      return
    }
    val thetaFname = args(0)
    val paramName = args(1)
    val outFname = args(2)
    val transpose = args(3).toBoolean
    val builder = new BuildArch()
    val theta = builder.loadTheta(thetaFname)
    println(" > Extracting "+paramName + " from: "+thetaFname)
    val param = theta.getParam(paramName)
    println(" > Saving "+paramName + " to: "+outFname)
    if(transpose)
      HMat.saveMat(outFname,param.t)
    else
      HMat.saveMat(outFname,param)
  }
}
