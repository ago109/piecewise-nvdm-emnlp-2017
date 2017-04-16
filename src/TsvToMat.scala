import scala.runtime.{RichInt, RichInt$}
//Imports from BIDMat
import BIDMat.{CMat, CSMat, DMat, Dict, FMat, GIMat, GMat, GSMat, HMat, IDict, IMat, Mat, SBMat, SDMat, SMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._

/**
  * Created by ago109 on 4/14/17.
  */
object TsvToMat {
  Mat.checkMKL

  def main(args : Array[String]): Unit ={
    val in = "h5nvdm/doc_vecs.txt" //args(0)
    val out = "h5nvdm/doc_vecs.mat" //args(1)

    val mat = loadFMat(in).t
    HMat.saveMat(out,FMat(mat))


  }


}
