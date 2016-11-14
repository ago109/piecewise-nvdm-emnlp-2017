import BIDMat.SciFunctions._
import YADLL.FunctionGraph.{BuildArch, Theta}
import YAVL.Data.Text.Lexicon.Lexicon

import scala.runtime.RichInt
//Imports from BIDMat
import BIDMat.{CMat, CSMat, DMat, Dict, FMat, GIMat, GMat, GSMat, HMat, IDict, IMat, Mat, SBMat, SDMat, SMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._

/**
  * Created by ago109 on 11/14/16.
  */
object EmbeddingAnalyzer {
  Mat.checkMKL //<-- needed in order to check for BIDMat on cpu or gpu...

  def norm2(mat:Mat):Mat={
    val z = abs(mat)
    return sqrt(sum(sum(z *@ z)))
  }

  def cosineSim(a : Mat, b : Mat):Float={
    val x_norm = norm2(a)
    val y_norm = norm2(b)
    return (a ^* b / (x_norm *@ y_norm)).dv.toFloat //obtain similarity score
  }

  def euclideanDist(a : Mat, b : Mat):Float={
    var delta = (b - a)
    delta = delta *@ delta //square of distance
    return sqrt( sum(sum(delta)) ).dv.toFloat
  }


  def main(args : Array[String]): Unit = {
    if (args.length < 7) {
      System.err.println("usage: [/path/to/theta] [decoder_name] [/path/to/dict] [top_k_symbols]" +
        " [performRot?] [querySymbol] [metric] ?[transposeDecoder?]")
      return
    }
    val archBuilder = new BuildArch()
    val theta = archBuilder.loadTheta(args(0))
    val decoderName = args(1)
    val dict = new Lexicon(args(2))
    val k = args(3).toInt
    val performRot = args(4).toBoolean
    val querySymbol = args(5)
    val query_idx = dict.getIndex(querySymbol)
    val metric = args(6)
    var transposeDecoder = false
    if(args.length >= 8){
      transposeDecoder = args(7).toBoolean
    }

    //Begin analysis
    var decoder = theta.getParam(decoderName) //terms are along the rows, latents along the columns
    if(transposeDecoder){
      println(" > Transposing decoder...")
      decoder = decoder.t
    }
    if(performRot){
      println(" > Rotating decocder...")
      val A = decoder * decoder.t
      val stat = seig(A,true)
      val eigens = stat._2.asInstanceOf[Mat] //get eigen-vectors
      decoder =eigens * decoder //rotate decoder
    }
    //Transpose decoder to column-major for efficient slicing
    decoder = decoder.t //terms along cols now, latents along rows

    //Get query embedding from decoder
    val query = decoder(?,query_idx)

    //Create an array of scores (1 for each embedding in decoder)
    //val scores = new Array[Float](decoder.ncols)
    println(" > Scoring terms based on decoder embeddings...")
    var scores:Mat = null
    var i = 0
    while(i < decoder.ncols){
      val target = decoder(?,i)
      var score = -1f
      if(metric.compareTo("cosine") == 0){
        score = EmbeddingAnalyzer.cosineSim(query,target)
      }else{ //else, Euclidean distance
        score = EmbeddingAnalyzer.euclideanDist(query,target)
      }
      if(null != scores){
        scores = scores on score
      }else{
        scores = score
      }
      i += 1
    }
    println(" > Sorting scores based on metric: "+metric)
    //We now have a column-vector of scores, these need to be sorted
    var sortedInd:IMat = null
    if(metric.compareTo("cosine") == 0){
      val stat = sortdown2(scores) //highest/most positive values at top
      scores = stat._1.asInstanceOf[Mat]
      sortedInd = stat._2.asInstanceOf[IMat]
    }else{ //Euclidean distance
      val stat = sort2(scores) //lowest values at top
      scores = stat._1.asInstanceOf[Mat]
      sortedInd = stat._2.asInstanceOf[IMat]
    }
    println(" > Extracting ranked "+metric+" scores...")
    var out = ""
    var k_i = 0
    var addOne = false
    while(k_i < k){
      val idx = sortedInd(k_i,0)
      val score = FMat(scores)(k_i,0)
      val symbol = dict.getSymbol(idx)
      if(symbol.compareTo(querySymbol) != 0){
        out += symbol + " : " + score
      }else{
        addOne = true //switch flag, since we have to add an extra score item to skip same word eval...
      }
      k_i += 1
    }
    if(addOne){
      val idx = sortedInd(k_i,0)
      val score = FMat(scores)(k_i,0)
      val symbol = dict.getSymbol(idx)
      out += symbol + " : " + score + "\n"
    }
    println(" ==== Query Results for "+querySymbol + " ==== \n\n" +out)
  }
}
