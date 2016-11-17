//General imports
import java.io.File
import java.util.HashMap
import java.util.ArrayList
import java.util.Random

import scala.io.StdIn.{readInt, readLine}
import BIDMat.FMat
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import YADLL.FunctionGraph.Graph.OGraph
import YADLL.FunctionGraph.Operators.SpecOps.{KL_Gauss, KL_Piece}
import YADLL.FunctionGraph.Optimizers.{SGOpt, _}
import YADLL.FunctionGraph.Theta
import YADLL.Utils.{MathUtils, MiscUtils}
import YAVL.Data.Text.Lexicon.Lexicon
import YAVL.Utils.{Logger, ScalaDebugUtils}

import scala.runtime.{RichInt, RichInt$}
//Imports from BIDMat
import BIDMat.{CMat, CSMat, DMat, Dict, FMat, GIMat, GMat, GSMat, HMat, IDict, IMat, Mat, SBMat, SDMat, SMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
//Imports from the library YADLL v0.7
import YADLL.FunctionGraph.BuildArch
import YADLL.Utils.ConfigFile

/**
  * Computes gradients of words w/ respect to Kl terms (either G-KL, P-KL, or both)
  * and ranks based on strength of gradients (i.e., L2-norms).
  *
  * Created by ago109 on 11/15/16.
  */
object WordSensitivityAnalyzer {
  Mat.checkMKL //<-- needed in order to check for BIDMat on cpu or gpu...

  /**
    * Converts a manual prompt to BOW under current dictionary.  Make sure
    * you use words in the lexicon only...otherwise, they will be discarded.
    * Note that routine automatically lower-cases prompts.
    *
    * @param prompt
    * @param dict
    * @return
    */
  def promptToBow(prompt : String, dict : Lexicon):DocSample ={
    val dim = dict.getLexiconSize()
    val tok = prompt.toLowerCase().split("\\s+") //<-- split at spaces
    val tokMap = new HashMap[Int,java.lang.Float]()
    var t = 0
    while(t < tok.length){
      val symb = tok(t)
      val symb_idx = dict.getIndex(symb)
      if(symb_idx != null){
        var freq = tokMap.get(symb_idx)
        if(freq != null){
          freq += 1f
          tokMap.put(symb_idx,freq)
        }else{
          tokMap.put(symb_idx,1f)
        }
      }//else, discard word
      t += 1
    }
    val bagOfIdx = new Array[Int](tokMap.size())
    val bagOfVals = new Array[Float](tokMap.size())
    val iter = (tokMap.keySet()).iterator()
    var ptr = 0
    while (iter.hasNext) {
      val idx = iter.next()
      val freq = tokMap.get(idx)
      bagOfIdx(ptr) = idx
      bagOfVals(ptr) = freq
      ptr += 1
    }
    /*
    //Generate x (or BOW)
    val x_ind = new IMat(1,bagOfIdx.length,bagOfIdx)
    val x_val = new FMat(1,bagOfVals.length,bagOfVals)
    //val x_col_i = (new RichInt(0) until this.bagOfIdx.length).toArray[Int]
    //val x_col = new IMat(1,x_col_i.length,x_col_i)
    val x_col = izeros(1,bagOfIdx.length)
    val x = sparse(IMat(x_ind),IMat(x_col),FMat(x_val),dim,1)
    return x
    */
    val doc = new DocSample(-1,dim,bagOfIdx,bagOfVals)
    return doc
  }

  //blockDerivFlow
  def main(args : Array[String]): Unit ={
    if(args.length != 1){
      System.err.println("usage: [/path/to/config.cfg]")
      return
    }
    //Extract arguments
    val configFile = new ConfigFile(args(0)) //grab configuration from local disk
    val seed = configFile.getArg("seed").toInt
    setseed(seed) //controls determinism of overall simulation
    val rng = new Random(seed)
    val dataFname = configFile.getArg("dataFname")
    val dictFname = configFile.getArg("dictFname")
    val graphFname = configFile.getArg("graphFname")
    val thetaFname = configFile.getArg("thetaFname")
    val labelFname = configFile.getArg("labelFname")
    val top_k = configFile.getArg("top_k").toInt
    var lookAtInputDeriv = configFile.getArg("lookAtInputDeriv").toBoolean
    if(lookAtInputDeriv){
      println(" > Building scores based on deriv of KL wrt input units")
    }else{
      println(" > Building scores based on L2 norms of KL wrt input embeddings")
    }
    var labelMap:HashMap[Int,String] = null
    if(configFile.isArgDefined("labelMapFname")){
      labelMap = new HashMap[Int,String]()
      val labelMapFname = configFile.getArg("labelMapFname")
      val lfd = MiscUtils.loadInFile(labelMapFname)
      var line = lfd.readLine()
      while(line != null){
        val tok = line.split("\\s+")
        labelMap.put(tok(1).toInt,tok(0))
        line = lfd.readLine()
      }
    }

    //Load in labels (1 per doc)
    val sorted_ptrs = new HashMap[Int,ArrayList[Int]]()
    val fd_lab = MiscUtils.loadInFile(labelFname)
    var line = fd_lab.readLine()
    var numRead = 0
    var idx = 0
    while(line != null){
      val lab = line.replaceAll(" ","").toInt
      var dat = sorted_ptrs.get(lab)
      if(dat != null){
        dat.add(idx)
      }else{
        dat = new ArrayList[Int]()
        dat.add(idx)
      }
      sorted_ptrs.put(lab,dat)
      numRead += 1
      print("\r > "+numRead + " labels read...")
      line = fd_lab.readLine()
      idx += 1
    }
    println()

    //Prop up sampler/lexicon
    val dict = new Lexicon(dictFname)
    val sampler = new DocSampler(dataFname,dict)
    sampler.loadDocsFromLibSVMToCache()

    //Load graph given config
    val archBuilder = new BuildArch()
    println(" > Loading Theta: "+thetaFname)
    val theta = archBuilder.loadTheta(thetaFname)
    println(" > Loading OGraph: "+graphFname)
    val graph = archBuilder.loadOGraph(graphFname)
    graph.theta = theta
    graph.hardClear() //<-- clear out any gunked up data from previous sessions

    //we need to know # of latent variables in model
    var n_lat = graph.getOp("z").dim
    if(graph.getOp("z-gaussian") != null){
      n_lat = graph.getOp("z-gaussian").dim
    }else if(graph.getOp("z-piece") != null){
      n_lat = graph.getOp("z-piece").dim
    }

    var lab_select = -4
    var manualDoc:DocSample = null
    while(lab_select != -1 || manualDoc != null){ //infinite loop unless user inputs -1 or enters a manual doc
      var docPtr = -1
      if(lab_select >= 0){
        val dat = sorted_ptrs.get(lab_select)
        if(dat.size() > 0){
          val s = MiscUtils.genRandInt(rng,0,dat.size())
          docPtr = dat.remove(s)
          sorted_ptrs.put(lab_select,dat)
        }else{
          println("  NO more data for "+lab_select)
        }
      }
      if(docPtr >= 0 || manualDoc != null){
        if(labelMap != null && lab_select >= 0){
          println(" >> LABEL => "+labelMap.get(lab_select))
        }
        var doc:DocSample = null
        if(manualDoc != null){
          doc = manualDoc
        }else
          doc = sampler.getDocAt(docPtr)
        println(" DOC:: "+doc.toDocString(dict))
        //Build an index look-up table for current document
        val idxLookUp = new HashMap[Int,java.lang.Float]()
        var ii = 0
        while(ii < doc.bagOfIdx.length){
          idxLookUp.put(doc.bagOfIdx(ii),doc.bagOfVals(ii))
          ii += 1
        }
        val x = doc.getBOWVec()
        //val x_i = sparse(w_idx,0,1f,doc.dim,1) //create a useless y or target
        val x_0 = sparse(0,0,1f,doc.dim,1) //create a useless y or target
        if(lookAtInputDeriv)
          graph.getOp("x-in").muteDerivCalc(flag = false) //we want the derivative w/ respect to word inputs
        //Generate random sample for model as needed
        val eps_gauss: Mat = ones(n_lat,x.ncols) *@ normrnd(0f, 1f, n_lat, 1)
        val eps_piece: Mat = ones(n_lat,x.ncols) *@ rand(n_lat, 1)
        val numDocs = 1 //<-- all of the predictions are for a single document
        val KL_correction:Mat = ones(1,x.ncols) *@ x.ncols //correct for KL being copied equal to # of predictions

        var output = "" //Symbol\tG-KL\tP-KL\n"
        val gkl = new HashMap[String,Float]()
        val pkl = new HashMap[String,Float]()

       //We call the inference routine eval() inside each potential KL-term block, since will need to tighten their bounds :)
        val KL_gauss_op = graph.getOp("KL-gauss")
        val KL_piece_op = graph.getOp("KL-piece")
        if(KL_gauss_op != null){
          val trickBak = KL_gauss_op.asInstanceOf[KL_Gauss].maxTrickConstant
          KL_gauss_op.asInstanceOf[KL_Gauss].maxTrickConstant = 0f
          graph.clamp(("x-in", x))
          graph.clamp(("x-targ", x_0))
          graph.clamp(("eps-gauss", eps_gauss))
          graph.clamp(("eps-piece", eps_piece))
          graph.clamp(("KL-correction",KL_correction))
          graph.clamp(("N",numDocs)) //<-- note we don't want a mean loss this time...
          graph.eval() //do inference to gather statistics
          println(" -> G-KL = "+graph.getStat("KL-gauss"))
          println(" Top("+top_k+") Most Sensitive to G-KL:")
          //Block gradient flow at the latent variable e
          graph.blockDerivFlow(graph.getOp("z"))
          if(KL_piece_op != null)
            graph.blockDerivFlow(KL_piece_op)
          val nabla = graph.calc_grad()
          var wordScores:Mat = null
          if(lookAtInputDeriv){
            wordScores = nabla.getParam("x-in") //get derivative of each term w/ respect to input words
            wordScores = wordScores *@ wordScores //squaring the square-root gives us squared L2 norm
            wordScores = wordScores / sum(sum(wordScores))
          }else{
            val d_e = nabla.getParam("W0-enc") //deriv wrt embeddings (i.e., slices in 1st encoder matrix)
            //println(ScalaDebugUtils.printFullMat(d_e))
            wordScores = sqrt( sum(d_e *@ d_e,1) ).t //L2 norm of each embedding deriv
            wordScores = wordScores *@ wordScores //we want squared L2-norm
            //Normalize scores to sum to 1.0
            wordScores = wordScores / sum(sum(wordScores)) //MathUtils.softmax(wordScores).asInstanceOf[Mat]
          }
          //Now build a ranked list based on gradients for each word in document
          val stat = sortdown2(wordScores) //highest/most positive values at top
          wordScores = stat._1.asInstanceOf[Mat]
          val sortedInd = stat._2.asInstanceOf[IMat]
          //With all scores sorted (and indices preserved), we may extract the ones in the document
          var  i = 0
          var k_found = 0
          while(i < wordScores.nrows && k_found < top_k){
            val score = FMat(wordScores)(i,0)
            val idx = sortedInd(i,0)
            if(idxLookUp.get(idx) != null){
              gkl.put(dict.getSymbol(idx),score)
              //output += dict.getSymbol(idx) + "\t"+score
              println(k_found + " -> " + dict.getSymbol(idx) + " = "+score)
              k_found += 1
            }
            i += 1
          }
          graph.unblockDerivs()
          KL_gauss_op.asInstanceOf[KL_Gauss].maxTrickConstant = trickBak
        }
        graph.hardClear()
        if(KL_piece_op != null){
          val trickBak = KL_piece_op.asInstanceOf[KL_Piece].maxTrickConstant
          KL_piece_op.asInstanceOf[KL_Piece].maxTrickConstant = 0f
          graph.clamp(("x-in", x))
          graph.clamp(("x-targ", x_0))
          graph.clamp(("eps-gauss", eps_gauss))
          graph.clamp(("eps-piece", eps_piece))
          graph.clamp(("KL-correction",KL_correction))
          graph.clamp(("N",numDocs)) //<-- note we don't want a mean loss this time...
          graph.eval() //do inference to gather statistics
          println(" -> P-KL = "+graph.getStat("KL-piece"))
          println(" Top("+top_k+") Most Sensitive to P-KL:")
          //Block gradient flow at the latent variable z!
          graph.blockDerivFlow(graph.getOp("z"))
          if(KL_gauss_op != null)
            graph.blockDerivFlow(KL_gauss_op)
          val nabla = graph.calc_grad()
          //Now build a ranked list based on gradients for each word in document
          var wordScores:Mat = null
          if(lookAtInputDeriv){
            wordScores = nabla.getParam("x-in") //get derivative of each term w/ respect to input words
            wordScores = wordScores *@ wordScores //we want squared L2-norm
            wordScores = wordScores / sum(sum(wordScores))
          }else{
            val d_e = nabla.getParam("W0-enc") //deriv wrt embeddings (i.e., slices in 1st encoder matrix)
            wordScores = sqrt( sum(d_e *@ d_e,1) ).t //L2 norm of each embedding deriv
            wordScores = wordScores *@ wordScores //we want squared L2-norm
            wordScores = wordScores / sum(sum(wordScores)) //MathUtils.softmax(wordScores).asInstanceOf[Mat]
          }
          //Now build a ranked list based on gradients for each word in document
          val stat = sortdown2(wordScores) //highest/most positive values at top
          wordScores = stat._1.asInstanceOf[Mat]
          val sortedInd = stat._2.asInstanceOf[IMat]
          //With all scores sorted (and indices preserved), we may extract the ones in the document
          var  i = 0
          var k_found = 0
          while(i < wordScores.nrows && k_found < top_k){
            val score = FMat(wordScores)(i,0)
            val idx = sortedInd(i,0)
            if(idxLookUp.get(idx) != null){
              pkl.put(dict.getSymbol(idx),score)
              println(k_found + " -> " + dict.getSymbol(idx) + " = "+score)
              k_found += 1
            }
            i += 1
          }
          graph.unblockDerivs()
          KL_piece_op.asInstanceOf[KL_Piece].maxTrickConstant = trickBak
        }
        graph.hardClear()
        //Compose output block
        val iter = (gkl.keySet()).iterator()
        val ww = new Array[String](gkl.size())
        val gg = new Array[Float](gkl.size())
        val pp = new Array[Float](gkl.size())
        var ptr = 0
        while (iter.hasNext) {
          val word = iter.next()
          val g_score = gkl.get(word)
          val p_score = pkl.get(word)
          ww(ptr) = word
          gg(ptr) = g_score
          pp(ptr) = p_score
          ptr += 1
          /*
          output += word + "\t" + g_score + "\t"
          if(KL_piece_op != null){
            output += ""+ p_score + "\n"
          }else{
            output += "0\n"
          }
          */
        }
        ptr = 0
        while(ptr < ww.length){
          output += ww(ptr)
          if(ptr != ww.length-1)
            output += " "
          ptr += 1
        }
        output += "\n"
        ptr = 0
        while(ptr < gg.length){
          output += gg(ptr)
          if(ptr != gg.length-1)
            output += " "
          ptr += 1
        }
        output += "\n"
        ptr = 0
        while(ptr < pp.length){
          output += pp(ptr)
          if(ptr != pp.length-1)
            output += " "
          ptr += 1
        }
        println("\n"+println(output))
        if(manualDoc != null)
          manualDoc = null
      }
      if(lab_select == -3){
        print(" > Enter a prompt: ")
        val prompt = readLine(" > Enter a custom doc:  ")
        manualDoc = WordSensitivityAnalyzer.promptToBow(prompt,dict)
        lab_select  = -4
      }else {
        print(" > Enter a label-index (-3 for writer, -1 to quit): ")
        lab_select = readInt()
        println()
        if (lab_select == -2) {
          if (lookAtInputDeriv)
            lookAtInputDeriv = false
          else
            lookAtInputDeriv = true
          println(" > Switching <lookAtInputDeriv> to " + lookAtInputDeriv)
        }
      }
    }
  }


}
