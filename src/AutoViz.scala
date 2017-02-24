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
  * Trains non-linear PCA to project an input file to a low-dimensional viz output.
  * (auto-associator based approach)
  *
  * arch := 40-30-...-20-k where k is final top-level dim
  *
  * Created by ago109 on 2/24/17.
  */
object AutoViz {
  Mat.checkMKL //<-- needed in order to check for BIDMat on cpu or gpu...

  def main(args : Array[String]): Unit = {
    if (args.length < 8) {
      System.err.println("usage: [/path/to/col-data.mat] [/path/to/outfname] [hid#-hid#...]" +
        " [hid-activation] [max_iter] [lr] [initType] [converge_eps]")
      return
    }
    val dat = HMat.loadFMat(args(0)) //<-- get column-major matrix of latent var vectors
    val outfname = args(1)
    val arch = args(2)
    val activation = args(3)
    val numIter = args(4).toInt
    val lr = args(5).toFloat
    val init = args(6)
    val converge_eps = args(7).toFloat
    val n_in = dat.nrows
    val n_out = dat.nrows
    val tok = arch.split("-")
    val hids = new Array[Int](tok.length)
    var i = 0
    while(i < tok.length){
      hids(i) = tok(i).toInt
      i += 1
    }
    //Start greedily building auto-encoders (stacked)
    i = 0
    var n_in_i = n_in
    var n_out_i = n_out
    var dat_project : Mat = dat
    while(i < hids.length){
      val n_hid = hids(i)
      if(i > 0){
        n_in_i = n_hid(i-1)
        n_out_i = n_in_i
      }
      val builder = new BuildArch()
      builder.initType = init
      builder.n_in = n_in_i
      builder.n_out = n_out_i
      builder.n_hid = n_hid
      builder.hidActivation = activation
      builder.outputActivation = "identity"
      val opt = new ADAM(lr = lr, opType = "descent")
      val aa = builder.buildAA()
      //Now train currently AA
      var iter = 0
      var notConverged = true
      while(iter < numIter && notConverged){
        //Gen permutation of sample indices
        val ind = randperm(dat_project.ncols)
        var L_prev = 1000000f
        var s = 0
        while(s < ind.ncols){
          val samp = dat_project(?,ind(0,s).dv.toInt)
          aa.clamp(("x0",samp),("y0",samp),("N",samp.ncols))
          aa.eval()
          val grad = aa.calc_grad()
          //Update model parameters using gradient
          opt.update(theta = aa.theta, nabla = grad)
          aa.hardClear()
          s += 1
        }
        //Evaluate current layer loss
        aa.clamp(("x0",dat_project),("y0",dat_project),("N",dat_project.ncols))
        aa.eval()
        val L_curr = aa.getStat("L").dv.toFloat
        println(iter + " Lyr = " + i + " Loss = " + L_curr)
        aa.hardClear()
        if((L_prev - L_curr) <= converge_eps){
          notConverged = false
        }
        L_prev = L_curr
        iter += 1
      }
      //Project data up to next level
      aa.clamp(("x0",dat_project),("y0",dat_project),("N",dat_project.ncols))
      aa.eval()
      dat_project = aa.getStat("h0")
      i += 1
    }
    println(" > Reached top of projection...saving "+hids(hids.length)+"-dim embeddings:")
    println("   "+outfname)
    HMat.saveFMatTxt(outfname,FMat(dat_project),delim = "\t")


  }

}
