import scala.runtime.RichInt
//Imports from BIDMat
import BIDMat.{CMat, CSMat, DMat, Dict, FMat, GIMat, GMat, GSMat, HMat, IDict, IMat, Mat, SBMat, SDMat, SMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import YADLL.Utils.ConfigFile
import YAVL.TextStream.Dict.Lexicon

/**
  * Created by ago109 on 1/31/17.
  */
object LatentVarAnalyzer {
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

  def main(args : Array[String]): Unit ={
    if (args.length < 5) {
      System.err.println("usage: [/path/to/lexicon.dict] [/path/to/latentVarFile] [metric] [top_k] [queryTerm]" +
        " [showScores?]")
      return
    }
    //Extract arguments
    val dictFname = args(0)
    val dict = new Lexicon(dictFname,true) //<-- uses deprecated constructor for now
    val latentVarFname = args(1)
    val latents = HMat.loadFMat(latentVarFname) //<-- get column-major matrix of latent var vectors
    val metric = args(2)
    val k = args(3).toInt
    val querySymbol = args(4)
    val query_idx = dict.getIndex(querySymbol)
    var showScores = false
    if(args.length >= 6){
      showScores = args(5).toBoolean
    }

    //Get latent vector
    val query = latents(?,query_idx)

    //Calculate cosine of this vector with every other vector in latent matrix
    println(" > Scoring terms based on decoder embeddings...")
    var scores:Mat = null
    var i = 0
    while(i < latents.ncols){
      val target = latents(?,i)
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
        if(showScores)
          out += symbol + " : " + score + "\n"
        else
          out += symbol + "\n"
      }else{
        addOne = true //switch flag, since we have to add an extra score item to skip same word eval...
      }
      k_i += 1
    }
    if(addOne){
      val idx = sortedInd(k_i,0)
      val score = FMat(scores)(k_i,0)
      val symbol = dict.getSymbol(idx)
      if(showScores)
        out += symbol + " : " + score + "\n"
      else
        out += symbol + "\n"
    }
    println(" ==== Query Results for "+querySymbol + " ==== \n" +out)
  }

}
