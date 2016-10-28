//General imports
import java.io.File
import java.util.ArrayList
import java.util.Random

import YAVL.Data.Text.Lexicon.Lexicon
import YAVL.Utils.ScalaDebugUtils

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

  def evalModel(rng : Random, x : Mat, y : Mat):Float ={

    return -1f
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

    //Build a sampler for main data-set
    val sampler = new DocSampler(dataFname,dict)
    sampler.loadDocsFromTSVToCache()

    val trainModel = configFile.getArg("trainModel").toBoolean
    if(trainModel){ //train/fit model to data
      val numEpochs = configFile.getArg("numEpochs").toInt
      val validFname = configFile.getArg("validFname")
      val errorMark = configFile.getArg("errorMark").toInt
      //Build validation set to conduct evaluation
      var validSampler = new DocSampler(validFname,dict)
      val dataChunks = Train.buildDataChunks(rng,validSampler,miniBatchSize)
      validSampler = null //toss aside this sampler for garbage collection

      //Build Ograph given config


      //Actualy train model
      var mark = 1
      var epoch = 0
      while(epoch < numEpochs) {
        var numSampsSeen = 0
        while (sampler.isDepleted() == false) {
          val batch = sampler.drawMiniBatch(rng, miniBatchSize)
          val x = batch._1.asInstanceOf[Mat]
          val y = batch._2.asInstanceOf[Mat]
          numSampsSeen += x.ncols

          if(numSampsSeen >= (mark * errorMark)){ //eval model @ this point

            mark += 1
          }

        }
        //Eval model after an epoch


        epoch += 1
        sampler.reset()
      }

    }else{ //evaluation only
      val graphFname = configFile.getArg("graphFname")
      val dataChunks = Train.buildDataChunks(rng,sampler,miniBatchSize)
      val N = sampler.numDocs() * 1f
      //Load graph given config
      val graph = archBuilder.loadOGraph(graphFname)
      graph.toggleOpOptimizations(false)
      //graph.muteEvals(true,"L") //avoid calculating outermost-loss

      var nll:Mat = 0f
      var i = 0
      while(i < dataChunks.size()){
        val batch = dataChunks.get(i)
        val x = batch._1.asInstanceOf[Mat]
        val y = batch._2.asInstanceOf[Mat]
        val L_n = sum((x > 0f),1) //get per-document lengths

        //Evaluate model on x using y as the target
        graph.clamp(("x-in",x))
        graph.clamp(("x-out",y))

        //Generate random samples for model is needed
        if(graph.modelTypeName.contains("hybrid")){
          val eps_gauss:Mat = normrnd(0f,1f,archBuilder.n_hid,x.ncols)
          val eps_piece:Mat = rand(archBuilder.n_hid,x.ncols)
          graph.clamp(("eps-gauss",eps_gauss))
          graph.clamp(("eps-piece",eps_piece))
        }else if(graph.modelTypeName.contains("gaussian")){
          val eps_gauss:Mat = normrnd(0f,1f,archBuilder.n_hid,x.ncols)
          graph.clamp(("eps-gauss",eps_gauss))
          graph.clamp(("x-out",y))
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
          }else if(graph.modelTypeName.contains("gaussian")){
            KL_term = graph.getOp("KL-gauss").per_samp_result
          }else if(graph.modelTypeName.contains("piece")){
            KL_term = graph.getOp("KL-piece").per_samp_result
          }
          val vlb = ln(P_theta) - KL_term //variational lower bound log P(X) = (ln P_Theta - KL)
          log_probs = vlb //user vlb in place of intractable distribution
        }else{  //If model is NOT variational, use its decoder's posterior
          log_probs = graph.getStat("L")
        }
        log_probs = log_probs *@ L_n //1/L_n * log P(X_n) for specific words in Doc_n
        nll += sum(log_probs)
        i += 1
      }
      nll = -nll / (N)
      val ppl = exp(nll)
      println(" Corpus.NLL = "+nll)
      println(" Corpus.PPL = "+ppl)
    }
  }


}
