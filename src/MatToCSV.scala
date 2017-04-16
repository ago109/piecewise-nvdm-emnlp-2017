import scala.runtime.{RichInt, RichInt$}
//Imports from BIDMat
import BIDMat.{CMat, CSMat, DMat, Dict, FMat, GIMat, GMat, GSMat, HMat, IDict, IMat, Mat, SBMat, SDMat, SMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._

/**
  * Created by ago109 on 4/14/17.
  */
object MatToCSV {
  Mat.checkMKL

  def main(args : Array[String]): Unit ={
    val in = "h-R.mat" //args(0)
    val out = "h-dat.tsv" //args(1)

    val mat = HMat.loadMat(in).t
    HMat.saveFMatTxt(out,FMat(mat),delim = "\t")


  }


}
