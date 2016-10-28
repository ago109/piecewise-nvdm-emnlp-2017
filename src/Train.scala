//General imports
import java.io.File
import java.util.ArrayList
import java.util.Random

import BIDMat.SciFunctions._
import YADLL.FunctionGraph.Graph.OGraph
import YADLL.FunctionGraph.Optimizers.{SGOpt, _}
import YADLL.FunctionGraph.Theta
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
  * Created by ago109 on 10/27/16.
  */
object Train {
  Mat.checkMKL //<-- needed in order to check for BIDMat on cpu or gpu...

  def buildDataChunks(rng : Random, sampler : DocSampler, blockSize : Int):ArrayList[(Mat,Mat)]={
    val chunks = new ArrayList[(Mat,Mat)]()
    while(sampler.isDepleted() == false){
      val batch = sampler.drawMiniBatch(rng,blockSize)
      chunks.add(batch)
    }
    return chunks
  }

  /**
    *
    * @param rng
    * @param graph
    * @param archBuilder
    * @param numDocs
    * @param dataChunks
    * @return (doc.nll, loss, KL-gaussian-score, KL-piecewise-score)
    */
  def evalModel(rng : Random, graph : OGraph, archBuilder : BuildArch, numDocs : Int,
                dataChunks : ArrayList[(Mat,Mat)]):Array[Mat] ={
    val stats =new Array[Mat](4)
    var doc_nll:Mat = 0f
    var loss_nll:Mat = 0f
    var KL_gauss_score:Mat = 0f
    var KL_piece_score:Mat = 0f
    var i = 0
    while(i < dataChunks.size()){
      val batch = dataChunks.get(i)
      val x = batch._1.asInstanceOf[Mat]
      val y = batch._2.asInstanceOf[Mat]
      val L_n = sum((x > 0f),1) //get per-document lengths
      val numSamps = x.ncols * 1f

      //Evaluate model on x using y as the target
      graph.clamp(("x-in",x))
      graph.clamp(("x-targ",y))

      //Generate random samples for model is needed
      if(graph.modelTypeName.contains("hybrid")){
        val eps_gauss:Mat = normrnd(0f,1f,archBuilder.n_hid,x.ncols)
        val eps_piece:Mat = rand(archBuilder.n_hid,x.ncols)
        graph.clamp(("eps-gauss",eps_gauss))
        graph.clamp(("eps-piece",eps_piece))
      }else if(graph.modelTypeName.contains("gaussian")){
        val eps_gauss:Mat = normrnd(0f,1f,archBuilder.n_hid,x.ncols)
        graph.clamp(("eps-gauss",eps_gauss))
      }else if(graph.modelTypeName.contains("piece")){
        val eps_piece:Mat = rand(archBuilder.n_hid,x.ncols)
        graph.clamp(("eps-piece",eps_piece))
      }

      //Run inference & estimate posterior probabilities
      graph.eval()
      var log_probs:Mat = null
      if(graph.modelTypeName.contains("vae")){ //If model is variational, use lower-bound to get probs
        val P_theta = sum(graph.getStat("x-out") *@ graph.getStat("x-targ"),1) //P_\Theta
        var KL_term:Mat = null
        if(graph.modelTypeName.contains("hybrid")){
          val KL_gauss = graph.getOp("KL-gauss").per_samp_result
          val KL_piece = graph.getOp("KL-piece").per_samp_result
          KL_term = KL_gauss + KL_piece
          KL_gauss_score += graph.getStat("KL-gauss") *@ numSamps
          KL_piece_score += graph.getStat("KL-piece") *@ numSamps
          loss_nll += graph.getStat("L") *@ numSamps
        }else if(graph.modelTypeName.contains("gaussian")){
          KL_term = graph.getOp("KL-gauss").per_samp_result
          KL_gauss_score += graph.getStat("KL-gauss") *@ numSamps
          loss_nll += graph.getStat("L") *@ numSamps
        }else if(graph.modelTypeName.contains("piece")){
          KL_term = graph.getOp("KL-piece").per_samp_result
          KL_piece_score += graph.getStat("KL-piece") *@ numSamps
          loss_nll += graph.getStat("L") *@ numSamps
        }
        val vlb = ln(P_theta) - KL_term //variational lower bound log P(X) = (ln P_Theta - KL)
        log_probs = vlb //user vlb in place of intractable distribution
      }else{  //If model is NOT variational, use its decoder's posterior
        log_probs = graph.getStat("L") *@ numSamps
      }
      log_probs = log_probs / L_n //1/L_n * log P(X_n) for specific words in Doc_n
      doc_nll += sum(log_probs)
      graph.hardClear()
      i += 1
    }
    stats(0) = -doc_nll / (1f * numDocs)
    stats(1) = -loss_nll / (1f * numDocs)
    stats(2) = KL_gauss_score / (1f * numDocs)
    stats(3) = KL_piece_score  / (1f * numDocs)
    return stats
  }

  def main(args : Array[String]): Unit ={
    if(args.length != 1){
      System.err.println("usage: [/path/to/configFile.cfg]")
      return
    }
    val configFile = new ConfigFile(args(0)) //grab configuration from local disk
    val archBuilder = new BuildArch()
    //Get simulation meta-parameters from config
    val seed = configFile.getArg("seed").toInt
    setseed(seed) //controls determinism of overall simulation
    val rng = new Random(seed)
    val dataFname = configFile.getArg("dataFname")
    val dictFname = configFile.getArg("dictFname")
    val dict = new Lexicon(dictFname)
    val miniBatchSize = configFile.getArg("miniBatchSize").toInt
    val outputDir = configFile.getArg("outputDir")
    val dir = new File(outputDir)
    dir.mkdir() //<-- build dir on disk if it doesn't already exist...
    val trainModel = configFile.getArg("trainModel").toBoolean
    val graphFname = configFile.getArg("graphFname")

    //Build a sampler for main data-set
    val sampler = new DocSampler(dataFname,dict)
    sampler.loadDocsFromTSVToCache()

    if(trainModel){ //train/fit model to data
      archBuilder.readConfig(configFile)
      archBuilder.n_in = dict.getLexiconSize() //input/output-dim is = to |V|
      val numEpochs = configFile.getArg("numEpochs").toInt
      val validFname = configFile.getArg("validFname")
      val errorMark = configFile.getArg("errorMark").toInt
      val norm_rescale = configFile.getArg("norm_rescale").toFloat
      val optimizer = configFile.getArg("optimizer")
      var lr = configFile.getArg("lr").toFloat
      //Build validation set to conduct evaluation
      var validSampler = new DocSampler(validFname,dict)
      validSampler.loadDocsFromTSVToCache()
      val valid_N = validSampler.numDocs()
      val dataChunks = Train.buildDataChunks(rng,validSampler,miniBatchSize)
      validSampler = null //toss aside this sampler for garbage collection

      //Build Ograph given config
      val graph = archBuilder.autoBuildGraph()
      graph.saveOGraph(outputDir+graphFname)

      //Build optimizer given config
      var opt:Opt = null
      var climb_type = "descent"
      if(archBuilder.archType.contains("vae")){
        climb_type = "ascent" //training a VAE requires gradient ascent!!
      }
      if(optimizer.compareTo("rmsprop") == 0){
        opt = new RMSProp(lr=lr,opType=climb_type)
      }else if(optimizer.compareTo("adam") == 0){
        opt = new ADAM(lr=lr,opType=climb_type)
      }else if(optimizer.compareTo("nadam") == 0){
        opt = new NADAM(lr=lr,mu = 0.99f, opType=climb_type)
      }else if(optimizer.compareTo("radam") == 0){
        opt = new RADAM(lr=lr,opType=climb_type)
      }else if(optimizer.compareTo("adagrad") == 0){
        opt = new AdaGrad(lr=lr,opType=climb_type)
      }else{ //default is good ol' SGD
        opt = new SGOpt(lr=lr,opType=climb_type)
      }
      opt.norm_threshold = norm_rescale

      val logger = new Logger(outputDir + graph.modelTypeName+"_stat.log")
      logger.openLogger()
      logger.writeStringln("Epoch, Valid.NLL, Valid.PPL, KL-Gauss, KL-Piece")

      var stats = Train.evalModel(rng,graph,archBuilder,valid_N,dataChunks)
      var bestNLL = stats(0)
      var bestPPL = exp(bestNLL)
      println("-1 > NLL = "+bestNLL + " PPL = " + bestPPL + " L = "+stats(1) + " KL.G = "+stats(2) + " KL.P = "+stats(3))

      //Actualy train model
      var avg_update_time = 0f
      var mark = 1
      var epoch = 0
      while(epoch < numEpochs) {
        if(epoch == (numEpochs-1)){
          opt.setPolyakAverage()
        }
        var numSampsSeen = 0 // # samples seen w/in an epoch
        var numIter = 0
        var currNLL:Mat = bestNLL
        var currPPL:Mat = bestPPL
        while (sampler.isDepleted() == false) {
          val t0 = System.nanoTime()
          /* ####################################################
           * Gather & clamp data/samples to OGraph
           * ####################################################
           */
          val batch = sampler.drawMiniBatch(rng, miniBatchSize)
          val x = batch._1.asInstanceOf[Mat]
          val y = batch._2.asInstanceOf[Mat]
          val numSamps = x.ncols
          numSampsSeen += numSamps
          numIter += 1
          graph.clamp(("x-in",x))
          graph.clamp(("x-targ",y))

          //Generate random samples for model is needed
          if(graph.modelTypeName.contains("hybrid")){
            val eps_gauss:Mat = normrnd(0f,1f,archBuilder.n_hid,x.ncols)
            val eps_piece:Mat = rand(archBuilder.n_hid,x.ncols)
            graph.clamp(("eps-gauss",eps_gauss))
            graph.clamp(("eps-piece",eps_piece))
          }else if(graph.modelTypeName.contains("gaussian")){
            val eps_gauss:Mat = normrnd(0f,1f,archBuilder.n_hid,x.ncols)
            graph.clamp(("eps-gauss",eps_gauss))
          }else if(graph.modelTypeName.contains("piece")){
            val eps_piece:Mat = rand(archBuilder.n_hid,x.ncols)
            graph.clamp(("eps-piece",eps_piece))
          }

          /* ####################################################
           * Run inference under model given data/samples
           * ####################################################
           */
          graph.eval()

          /* ####################################################
           * Estimate parameter-gradients
           * ####################################################
           */
          val grad = graph.calc_grad()

          /* ####################################################
           * Update model given gradients
           * ####################################################
           */
          opt.update(theta = graph.theta, nabla = grad, miniBatchSize = numSamps)

          val t1 = System.nanoTime()
          avg_update_time += (t1 - t0)

          if(errorMark > 0 && numSampsSeen >= (mark * errorMark)){ //eval model @ this point
            stats = Train.evalModel(rng,graph,archBuilder,valid_N,dataChunks)
            currNLL = stats(0)
            //logger.writeStringln("Epoch, Valid.NLL, Valid.PPL, KL-Gauss, KL-Piece")
            currPPL = exp(currNLL)
            if(currNLL.dv.toFloat <= bestNLL.dv.toFloat){
              bestNLL = currNLL
              bestPPL = currPPL
              graph.theta.saveTheta(outputDir+"best_at_epoch_"+epoch)
            }
            mark += 1
            println("\n > NLL = "+currNLL + " PPL = " + currPPL + " T = "+ (avg_update_time/numIter * 1e-9f) + " s")
          }
          println("\r > NLL = "+currNLL + " PPL = " + currPPL + " T = "+ (avg_update_time/numIter * 1e-9f) + " s")
        }
        println()
        //Checkpoint save current \Theta of model
        graph.theta.saveTheta(outputDir+"check_epoch_"+epoch)

        var polyak_avg:Theta = null
        if(epoch == (numEpochs-1)){
          println(" >> Estimating Polyak average over Theta...")
          polyak_avg = opt.estimatePolyakAverage()
          polyak_avg.saveTheta(outputDir+"polyak_avg")
        }

        //Eval model after an epoch
        stats = Train.evalModel(rng,graph,archBuilder,valid_N,dataChunks)
        currNLL = stats(0)
        currPPL = exp(currNLL)
        println(epoch+" > NLL = "+currNLL + " PPL = " + currPPL+ " L = "+stats(1) + " KL.G = "+stats(2) + " KL.P = "+stats(3))
        epoch += 1
        sampler.reset()
      }
    }else{ //evaluation only
      val N = sampler.numDocs()
      val dataChunks = Train.buildDataChunks(rng,sampler,miniBatchSize)
      //Load graph given config
      val graph = archBuilder.loadOGraph(graphFname)
      graph.hardClear() //<-- clear out any gunked up data from previous sessions
      //graph.muteEvals(true,"L") //avoid calculating outermost-loss
      val stats = Train.evalModel(rng,graph,archBuilder,N,dataChunks)
      val nll = stats(0)
      val ppl = exp(nll)
      println(" ====== Performance ======")
      println(" > Corpus.NLL = "+nll)
      println(" > Corpus.PPL = "+ppl)
      println(" > over "+N + " documents")
    }
  }


}
