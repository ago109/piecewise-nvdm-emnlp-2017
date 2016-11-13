import YADLL.FunctionGraph.{BuildArch, Theta}
import YAVL.Data.Text.Lexicon.Lexicon

import scala.runtime.RichInt
//Imports from BIDMat
import BIDMat.{CMat, CSMat, DMat, Dict, FMat, GIMat, GMat, GSMat, HMat, IDict, IMat, Mat, SBMat, SDMat, SMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._

/**
  * Created by ago109 on 11/11/16.
  */
object DecoderAnalyzer {
  Mat.checkMKL //<-- needed in order to check for BIDMat on cpu or gpu...

  def main(args : Array[String]): Unit ={
    if(args.length != 4){
      System.err.println("usage: [/path/to/theta] [decoder_name] [/path/to/dict] [top_k_symbols]")
      return
    }
    val archBuilder = new BuildArch()
    val theta = archBuilder.loadTheta(args(0))
    val decoderName = args(1)
    val dict = new Lexicon(args(2))
    val k = args(3).toInt

    //Begin analysis
    val decoder = theta.getParam(decoderName) //terms are along the rows, latents along the columns
    val cols_ind = new RichInt(0) until dict.getLexiconSize()
    val cols = new IMat(1,decoder.ncols,cols_ind.toArray[Int])

    println(" > Analyzing decoder agaisnt lexcion.size = "+dict.getLexiconSize())
    val topSymbols = new Array[Array[String]](decoder.ncols) //will have an array for each latent i
    var k_i = 0
    while(k_i < k) { //for
      val stat = maxi2(decoder, 1)
      val rows = stat._2.asInstanceOf[IMat]
      var vals = stat._1.asInstanceOf[FMat]
      //println("Rows: "+(rows))
      //println("Vals: "+(vals))
      val absMin = mini(mini(decoder)).dv.toFloat
      var q = 0
      while(q < rows.ncols){
        val r_idx = rows(0,q)
        val c_idx = q
        decoder(r_idx,c_idx) = (absMin - 1f)
        q += 1
      }

      //decoder(rows,cols) = (absMin -1f)
      //Get symbols w/ highest positive score w/ respect to each latent variable
      var i = 0
      while(i < rows.ncols){
        if(topSymbols(i) == null){
          topSymbols(i) = new Array[String](k) //init any empty slots
        }
        val symbol_idx = rows(0,i)
        val symbol = dict.getSymbol(symbol_idx) //get top symbol for latent variable i
        topSymbols(i)(k_i) = symbol // for symbol k_i insert in ranking for latent i
        i += 1
      }
      k_i += 1
    }
    //Print ranked list to standard out
    var out = ""
    var i = 0
    while(i < topSymbols.length) {
      out += "Latent("+i+"):  "
      k_i = 0
      while(k_i < topSymbols(i).length){
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
