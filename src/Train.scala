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
    //Get simulation meta-parameters from config
    val seed = configFile.getArg("seed").toInt
    setseed(seed) //controls determinism of overall simulation
    val rng = new Random(seed)
    val dataFname = configFile.getArg("dataFname")
    val dictFname = configFile.getArg("dictFname")
    val dict = new Lexicon(dictFname)
    val cacheSize = configFile.getArg("cacheSize").toInt
    val miniBatchSize = configFile.getArg("miniBatchSize").toInt
    //Build a sampler for main data-set
    val sampler = new DocSampler(dataFname,dict,cacheSize)
    sampler.loadDocsFromTSVToCache()

    val trainModel = configFile.getArg("trainModel").toBoolean
    if(trainModel){ //train/fit model to data
      val numEpochs = configFile.getArg("numEpochs").toInt
      val validFname = configFile.getArg("validFname")
      val errorMark = configFile.getArg("errorMark").toInt
      //Build validation set to conduct evaluation
      var validSampler = new DocSampler(validFname,dict,cacheSize)
      val dataChunks = Train.buildDataChunks(rng,validSampler,miniBatchSize)
      validSampler = null //toss aside this sampler for garbage collection

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
      val dataChunks = Train.buildDataChunks(rng,sampler,miniBatchSize)
      //TODO: mute calculation for outermost loss L in model...
      var i = 0
      while(i < dataChunks.size()){
        val batch = dataChunks.get(i)
        val x = batch._1.asInstanceOf[Mat]
        val y = batch._2.asInstanceOf[Mat]
        //Evaluate model on x using y as the target
        println("------------------------------")
        println("X:\n"+ScalaDebugUtils.printFullMat(x))
        println("Y:\n"+ScalaDebugUtils.printFullMat(y))

        //If model is variational, use its lower-bound to get probs


        //If model is NOT variational, use its decoder's posterior

        i += 1
      }

    }
  }


}
