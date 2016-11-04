import java.util.Random

import YAVL.Data.Text.Lexicon.Lexicon
import BIDMat.{CMat, CSMat, DMat, Dict, FMat, GIMat, GMat, GSMat, HMat, IDict, IMat, Mat, SBMat, SDMat, SMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import YAVL.Utils.ScalaDebugUtils

/**
  * Created by ago109 on 11/4/16.
  */
object SamplerTest {

  def main(args : Array[String]): Unit ={
    if(args.length < 4){
      System.err.println("usage: [seed] [dataFname] [dictFname] [batchsize]")
    }
    val seed = args(0).toInt
    val rng = new Random(seed)
    val dataFname = args(1)
    val dictFname = args(2)
    val batchSize = args(3).toInt
    val dict = new Lexicon(dictFname)
    val sampler = new DocSampler(dataFname,dict)
    sampler.loadDocsFromLibSVMToCache()
    println(" === Random Mini-Batch Samples === ")
    while(sampler.isDepleted() == false){
      val stat = sampler.drawMiniBatch(batchSize,rng)
      val x = stat._1.asInstanceOf[Mat]
      val y = stat._2.asInstanceOf[Mat]
      println("X.B:\n"+ScalaDebugUtils.printFullMat(x))
      println("Y.B:\n"+ScalaDebugUtils.printFullMat(y))
    }
    sampler.reset()
    println(" === Ordered Samples === ")
    while(sampler.isDepleted() == false){
      val stat = sampler.drawFullDocBatch()
      val x = stat._1.asInstanceOf[Mat]
      val y = stat._2.asInstanceOf[Mat]
      println("X.B:\n"+ScalaDebugUtils.printFullMat(x))
      println("Y.B:\n"+ScalaDebugUtils.printFullMat(y))
    }
    sampler.reset()
    println(" === Random Doc-Based Samples === ")
    while(sampler.isDepleted() == false){
      val stat = sampler.drawFullRandDoc(rng)
      val x = stat._1.asInstanceOf[Mat]
      val y = stat._2.asInstanceOf[Mat]
      println("X.B:\n"+ScalaDebugUtils.printFullMat(x))
      println("Y.B:\n"+ScalaDebugUtils.printFullMat(y))
    }

  }
}
