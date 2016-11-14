import YADLL.FunctionGraph.{BuildArch, Theta}
import YAVL.Data.Text.Lexicon.Lexicon

import scala.runtime.RichInt
//Imports from BIDMat
import BIDMat.{CMat, CSMat, DMat, Dict, FMat, GIMat, GMat, GSMat, HMat, IDict, IMat, Mat, SBMat, SDMat, SMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._

/**
  * Analyzes decoder (or rather, any conforming weight matrix) of a model assuming that its columns
  * (either w/ or w/o a tranpose) align to unique latent variables and its rows align with a symbol
  * lexicon.
  *
  * Created by ago109 on 11/11/16.
  */
object DecoderAnalyzer {
  Mat.checkMKL //<-- needed in order to check for BIDMat on cpu or gpu...

  def main(args : Array[String]): Unit ={
    if(args.length < 6){
      System.err.println("usage: [/path/to/theta] [decoder_name] [/path/to/dict] [top_k_symbols]" +
        " [useMax?] [performRot?] ?[showScores?] ?[transposeDecoder?]")
      return
    }
    val archBuilder = new BuildArch()
    val theta = archBuilder.loadTheta(args(0))
    val decoderName = args(1)
    val dict = new Lexicon(args(2))
    val k = args(3).toInt
    val useMax = args(4).toBoolean
    val performRot = args(5).toBoolean
    if(useMax){
      println(" > Using max-scored symbol weights")
    }else{
      println(" > Using min-scored symbol weights")
    }
    var showScores = false
    if(args.length >= 7){
      showScores = args(6).toBoolean
    }
    var transposeDecoder = false
    if(args.length >= 8){
      transposeDecoder = args(7).toBoolean
    }

    //Begin analysis
    var decoder = theta.getParam(decoderName) //terms are along the rows, latents along the columns
    if(transposeDecoder){
      decoder = decoder.t
    }

    //val cols_ind = new RichInt(0) until dict.getLexiconSize()
    if(performRot){
      println(" > Rotating decocder...")
      var A = decoder * decoder.t
      val stat = seig(A,true)
      val eigens = stat._2.asInstanceOf[Mat] //get eigen-vectors
      decoder =eigens * decoder //rotate decoder
    }
    //val cols = new IMat(1,decoder.ncols,cols_ind.toArray[Int])

    println(" > Analyzing decoder agaisnt lexcion.size = "+dict.getLexiconSize())
    val topSymbols = new Array[Array[String]](decoder.ncols) //will have an array for each latent i
    val topScores = new Array[Array[Float]](decoder.ncols)
    //Perform a sort of values in decoder
    var sortedInd:Mat = null
    if(useMax){
      val stat = sortdown2(decoder) //highest/most positive values at top
      decoder = stat._1.asInstanceOf[Mat]
      sortedInd = stat._2.asInstanceOf[Mat]
    }else{
      val stat = sort2(decoder) //lowest values at top
      decoder = stat._1.asInstanceOf[Mat]
      sortedInd = stat._2.asInstanceOf[Mat]
    }
    var l = 0
    while(l < decoder.ncols){
      val lat_scores = decoder(?,l) //grab latent scores
      val lat_ind = sortedInd(?,l) //grab indices (row)
      var ptr = 0
      while(ptr < k){ // lat_ind.nrows (note, we only go up to k-most entries in ranked list)
        val score = FMat(lat_scores)(ptr,0)
        val idx = IMat(lat_ind)(ptr,0)
        val symbol = dict.getSymbol(idx) //get top symbol for latent variable i
        if(topSymbols(l) == null){
          topSymbols(l) = new Array[String](k) //init any empty slots
          topScores(l) = new Array[Float](k)
        }
        topSymbols(l)(ptr) = symbol
        topScores(l)(ptr) = score
        ptr += 1
      }
      l += 1
    }
    //Print ranked list to standard out
    var out = ""
    var i = 0
    while(i < topSymbols.length) {
      out += "Latent("+i+"):  "
      var k_i = 0
      while(k_i < topSymbols(i).length){
        if(showScores)
          out += topSymbols(i)(k_i) + " (" + topScores(i)(k_i) + ") "
        else
          out += topSymbols(i)(k_i) + " "
        k_i += 1
      }
      out += "\n"
      i += 1
    }
    println(" ------------------------------ ")
    println(" ----- Top-Scored Symbols ----- ")
    println(out)
  }


}
