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
  * This is a simple bit of code meant to compute the latent variables (under a NVDM-type model)
  * for specific words in a given vocabulary.  Takes in a trained model and a lexicon (that was used in building
  * the document model) and extracts relevant NVDM's latent variables to disk...
  *
  * "About the visualization. Yes encoding each word into its latent representation and then looking at
  * neighbouring words would be one way to do it. It would be especially useful to illustrate multi-modal
  * aspects. For example, a verb like "shot" might appear both in political articles ( "soldiers shot at ...")
  * and in sports ("he shot three goals').An alternative to looking at nearest neighbours might be to
  * visualize these encodings using t-sne."
  *
  * Created by ago109 on 1/30/17.
  */
object LatentVarExtractor {
  Mat.checkMKL //<-- needed in order to check for BIDMat on cpu or gpu...

  def main(args : Array[String]): Unit ={
    if(args.length != 1){
      System.err.println("usage: [/path/to/config.cfg]")
      return
    }
    //Extract arguments
    val configFile = new ConfigFile(args(0)) //grab configuration from local disk
    val seed = configFile.getArg("seed").toInt
    setseed(seed) //controls determinism of overall simulation
    //val rng = new Random(seed)
    val dictFname = configFile.getArg("dictFname")
    val lex = new Lexicon(dictFname,true) //<-- uses deprecated constructor for now
    val graphFname = configFile.getArg("graphFname")
    val thetaFname = configFile.getArg("thetaFname")
    val gaussianVarFname = configFile.getArg("gaussianVarFname") //<--saves latent variables to disk
    val pieceVarFname = configFile.getArg("pieceVarFname") //<--saves latent variables to disk
    val latentVarFname = configFile.getArg("latentVarFname")
    //Load graph given config
    val archBuilder = new BuildArch()
    println(" > Loading Theta: "+thetaFname)
    val theta = archBuilder.loadTheta(thetaFname)
    println(" > Loading OGraph: "+graphFname)
    val graph = archBuilder.loadOGraph(graphFname)
    graph.theta = theta
    graph.hardClear() //<-- clear out any gunked up data from previous sessions

    val n_lat = graph.getOp("z").dim //final dim of latent variable (may be hybrid)
    var n_lat_p = -1 // # piecewise variables
    var n_lat_g = -1 // # gaussian variables
    if(graph.getOp("z-gaussian") != null){
      n_lat_g = graph.getOp("z-gaussian").dim
    }
    if(graph.getOp("z-piece") != null){
      n_lat_p = graph.getOp("z-piece").dim
    }

    //Now, loop through each word in lexicon, and calculate latent variable
    var gaussianMat : Mat = null
    var pieceMat : Mat = null
    var latentMat : Mat = null
    val lex_size = lex.getLexiconSize()
    var word_idx = 0
    while(word_idx < lex_size){
      val word = lex.getSymbol(word_idx) //<-- get actual word/symbol from lexicon given index
      print("\r Getting latents for "+word)
      val x_w = sparse(word_idx,0,1f,lex_size,1) //<-- create one-hot encoding for word
      //Generate some samples for latent variable(s)
      var eps_gauss: Mat = null
      var eps_piece: Mat = null
      val KL_correction:Mat = ones(1,x_w.ncols) *@ x_w.ncols
      if(n_lat_g > 0){
        eps_gauss = ones(n_lat_g,x_w.ncols) *@ normrnd(0f, 1f, n_lat_g, 1)
      }
      if(n_lat_p > 0){
        eps_piece = ones(n_lat_p,x_w.ncols) *@ rand(n_lat_p, 1)
      }
      //val KL_gauss_op = graph.getOp("KL-gauss")
      //val KL_piece_op = graph.getOp("KL-piece")
      graph.clamp(("x-in", x_w))
      graph.clamp(("x-targ", x_w)) //<-- we don't care about the target this time...
      graph.clamp(("eps-gauss", eps_gauss))
      graph.clamp(("eps-piece", eps_piece))
      graph.clamp(("KL-correction",KL_correction))
      graph.clamp(("N",x_w.ncols)) //<-- we don't care about the loss this time...
      graph.eval() //do inference to gather statistics
      //Extract relevant latent variables from NVDM-model
      if(n_lat_g > 0){
        val gaussVar = graph.getStat("z-gaussian")
        // write word and latent variable to gaussian-variable file
        if(null != gaussianMat){
          gaussianMat = gaussianMat \ gaussVar
        }else{
          gaussianMat = gaussVar
        }
      }
      if(n_lat_p > 0){
        val pieceVar = graph.getStat("z-piece")
        // write word and latent variable to piecewise-variable file
        if(null != pieceMat){
          pieceMat = pieceMat \ pieceVar
        }else{
          pieceMat = pieceVar
        }
      }
      val latVar = graph.getStat("z")
      // write hybrid latent variable and word to hybrid-variable file...
      if(null != latentMat){
        latentMat = latentMat \ latVar
      }else{
        latentMat = latVar
      }
      word_idx += 1
    }
    println()

    //Write composed column-major matrices to local disk
    println(" > Saving:  "+latentVarFname)
    HMat.saveMat(latentVarFname,latentMat)
    if(null != gaussianMat){
      println(" > Saving:  "+gaussianVarFname)
      HMat.saveMat(gaussianVarFname,gaussianMat)
    }
    if(null != pieceMat){
      println(" > Saving:  "+pieceVarFname)
      HMat.saveMat(pieceVarFname,pieceMat)
    }
  }


}
