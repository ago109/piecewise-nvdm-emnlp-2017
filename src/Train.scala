//General imports
import java.io.File
import java.util.ArrayList
import java.util.Random

import BIDMat.SciFunctions._
import YADLL.FunctionGraph.Graph.OGraph
import YADLL.FunctionGraph.Operators.SpecOps.{KL_Gauss, KL_Piece}
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

  //def buildDataChunks(rng : Random, sampler : DocSampler, blockSize : Int)
  def buildFullSample(sampler : DocSampler):ArrayList[(Mat,Mat)]={
    val chunks = new ArrayList[(Mat,Mat)]()
    while(sampler.isDepleted() == false){
      val batch = sampler.drawFullDocBatch() //drawMiniBatch(blockSize, rng)
      chunks.add(batch)
    }
    return chunks
  }

  def buildRandSample(rng : Random, sampler : DocSampler, numSamps : Int):ArrayList[(Mat,Mat)]={
    val chunks = new ArrayList[(Mat,Mat)]()
    var n = 0
    while(n < numSamps){
      val docBatch = sampler.drawFullRandDoc(rng)
      chunks.add(docBatch)
      n += 1
    }
    return chunks
  }

  //Some sub-routines for calculating the variational lower bound

  /**
    * Estimate a looser variational lower bound through sampling.
    *
    * @param rng
    * @param graph
    * @param x
    * @param y
    * @param n_lat
    * @param numModelSamples
    * @return
    */
  def getSampledBound(rng : Random, graph : OGraph, x : Mat, y : Mat, n_lat : Int, numModelSamples : Int = 10): (Mat,Mat,Mat,Mat) ={
    val L_n = x.ncols * 1f //get per-document lengths
    var s = 0
    var log_probs: Mat = null
    var KL_gauss_s:Mat = 0f
    var KL_piece_s:Mat = 0f
    while(s < numModelSamples) {
      //Evaluate model on x using y as the target
      graph.clamp(("x-in", x))
      graph.clamp(("x-targ", y))

      //Generate random sample for model as needed
      if (graph.modelTypeName.contains("hybrid")) {
        val eps_gauss: Mat = normrnd(0f, 1f, n_lat, x.ncols)
        val eps_piece: Mat = rand(n_lat, x.ncols)
        graph.clamp(("eps-gauss", eps_gauss))
        graph.clamp(("eps-piece", eps_piece))
      } else if (graph.modelTypeName.contains("gaussian")) {
        val eps_gauss: Mat = normrnd(0f, 1f, n_lat, x.ncols)
        graph.clamp(("eps-gauss", eps_gauss))
      } else if (graph.modelTypeName.contains("piece")) {
        val eps_piece: Mat = rand(n_lat, x.ncols)
        graph.clamp(("eps-piece", eps_piece))
      }
      graph.clamp(("N",1f)) //<-- note we don't want a mean loss this time...

      // Infer/estimate posterior probabilities
      var vlb:Mat = null
      graph.eval()
      if (graph.modelTypeName.contains("vae")) {
        //If model is variational, use lower-bound to get probs
        val P_theta = sum(graph.getStat("x-out") *@ graph.getStat("x-targ"), 1) //P_\Theta
        var KL_term: Mat = null
        if (graph.modelTypeName.contains("hybrid")) {
          if (null != graph.theta.getParam("gamma")) {
            val KL_gauss = graph.getOp("KL-gauss").per_samp_result
            val KL_piece = graph.getOp("KL-piece").per_samp_result
            KL_term = KL_gauss + KL_piece
            KL_gauss_s += graph.getStat("KL-gauss") //*@ numSamps
            KL_piece_s += graph.getStat("KL-piece") //*@ numSamps
          } else {
            KL_term = 0f
          }
        } else if (graph.modelTypeName.contains("gaussian")) {
          if (null != graph.theta.getParam("gamma")) {
            KL_term = graph.getOp("KL-gauss").per_samp_result
            KL_gauss_s += graph.getStat("KL-gauss") //*@ numSamps
          } else {
            KL_term = 0f
          }
        } else if (graph.modelTypeName.contains("piece")) {
          if (null != graph.theta.getParam("gamma")) {
            KL_term = graph.getOp("KL-piece").per_samp_result
            KL_piece_s += graph.getStat("KL-piece") //*@ numSamps
          } else {
            KL_term = 0f
          }
        }
        vlb = ln(P_theta) - KL_term //variational lower bound log P(X) = (ln P_Theta - KL)
      }

      //Now update our evaluation across data-set w/ found validation lower bound
      if (null != log_probs) {
        log_probs = log_probs + vlb
      } else
        log_probs = vlb //user vlb in place of intractable distribution

      s += 1
      if(graph.getOp("h1") != null){ //freeze most of the encoder, as it calcs doc rep only once!
        graph.toggleFreezeOp("h1",true)
      }else{
        graph.toggleFreezeOp("h0",true)
      }
    }
    log_probs = log_probs / (numModelSamples * 1f) // get E[VLB]
    val doc_nll = (sum(log_probs) / L_n) // <-- here we normalize by document lengths
    KL_gauss_s = (KL_gauss_s / (numModelSamples * 1f))
    KL_piece_s = (KL_piece_s / (numModelSamples * 1f))

    if(graph.getOp("h1") != null){
      graph.toggleFreezeOp("h1",false)
    }else{
      graph.toggleFreezeOp("h0",false)
    }
    graph.hardClear()

    return (log_probs,doc_nll,KL_gauss_s,KL_piece_s)

  }

  /**
    * Estimate a tighter lower variational bound using iterative inference (or SGD-inference).
    *
    * @param rng
    * @param graph
    * @param x
    * @param y
    * @param n_lat
    * @param numSGDInfSteps
    * @param lr_inf
    * @return
    */
  def getIterativeBound(rng : Random, graph : OGraph, x : Mat, y : Mat, n_lat : Int,
                        numSGDInfSteps : Int = 1, lr_inf : Float = 0.1f, lex : Lexicon = null ): (Mat,Mat,Mat,Mat) ={
    graph.muteDerivs(true,graph.theta)
    val L_n = x.ncols * 1f //get per-document lengths
    var log_probs: Mat = null
    var KL_gauss_s:Mat = 0f
    var KL_piece_s:Mat = 0f
    //Evaluate model on x using y as the target
    graph.clamp(("x-in", x)) //<-- BOW, document
    graph.clamp(("x-targ", y)) //<-- target word(s) in document

    //Generate random sample for model as needed
    if (graph.modelTypeName.contains("hybrid")) {
      val eps_gauss: Mat = normrnd(0f, 1f, n_lat, x.ncols)
      val eps_piece: Mat = rand(n_lat, x.ncols)
      graph.clamp(("eps-gauss", eps_gauss))
      graph.clamp(("eps-piece", eps_piece))
    } else if (graph.modelTypeName.contains("gaussian")) {
      val eps_gauss: Mat = normrnd(0f, 1f, n_lat, x.ncols)
      graph.clamp(("eps-gauss", eps_gauss))
    } else if (graph.modelTypeName.contains("piece")) {
      val eps_piece: Mat = rand(n_lat, x.ncols)
      graph.clamp(("eps-piece", eps_piece))
    }
    graph.clamp(("N",1f)) //<-- note we don't want a mean loss this time...

    // Infer/estimate posterior probabilities
    var vlb:Mat = null
    var step = 0
    while(step < numSGDInfSteps) {
      if(step > 0){ //for every step beyond the first, we perform an iterative inference step...
        //We have to carefully freeze the right parts of the graph to do proper partial SGD-inference
        if (graph.modelTypeName.contains("hybrid") || graph.modelTypeName.contains("gaussian")) {
          graph.toggleFreezeOp("mu-prior", true)
          graph.toggleFreezeOp("sigma-prior", true)
          graph.toggleFreezeOp("mu-post", true)
          graph.getOp("mu-post").muteEval(false) //to allow a gradient to be computed w/ respect to this
          graph.toggleFreezeOp("sigma-post", true)
          graph.getOp("sigma-post").muteEval(false) //to allow a gradient to be computed w/ respect to this
        }
        if (graph.modelTypeName.contains("hybrid") || graph.modelTypeName.contains("piece")) {
          val a_prior_name = "a-prior-"
          val a_post_name = "a-post-"
          var i = 0
          while(graph.getOp((a_prior_name + i)) != null){
            graph.toggleFreezeOp(a_prior_name+i,true)
            graph.toggleFreezeOp(a_post_name+i,true)
            graph.getOp(a_post_name+i).muteEval(false)//to allow a gradient to be computed w/ respect to this
            i += 1
          }
        }
        val grad = graph.calc_grad() //get gradients for non-frozen parts of graph
        //Now update the relevant parts of the graph using SGD
        if (graph.modelTypeName.contains("hybrid") || graph.modelTypeName.contains("gaussian")) {
          val mu_post_rho = grad.getParam("mu-post")
          val sigma_post_rho = grad.getParam("sigma-post")
          val mu_post = graph.getStat("mu-post")
          val sigma_post = graph.getStat("sigma-post")
          graph.getOp("mu-post").result = mu_post - (mu_post_rho *@ lr_inf)
          graph.getOp("sigma-post").result = sigma_post - (sigma_post_rho *@ lr_inf)
          graph.getOp("mu-post").muteEval(true)
          graph.getOp("sigma-post").muteEval(true)
        }
        if (graph.modelTypeName.contains("hybrid") || graph.modelTypeName.contains("piece")) {
          val a_post_name = "a-post-"
          var i = 0
          while(graph.getOp((a_post_name + i)) != null){
            val a_post_rho = grad.getParam(a_post_name + i)
            val a_post = graph.getStat(a_post_name+i)
            graph.getOp(a_post_name+i).result = a_post - (a_post_rho *@ lr_inf)
            graph.getOp(a_post_name+i).muteEval(true)
            i += 1
          }
        }
        //Now that posterior parameters have been update, we generate a fresh sample to re-compute z
        if (graph.modelTypeName.contains("hybrid")) {
          val eps_gauss: Mat = normrnd(0f, 1f, n_lat, x.ncols)
          val eps_piece: Mat = rand(n_lat, x.ncols)
          graph.clamp(("eps-gauss", eps_gauss))
          graph.clamp(("eps-piece", eps_piece))
        } else if (graph.modelTypeName.contains("gaussian")) {
          val eps_gauss: Mat = normrnd(0f, 1f, n_lat, x.ncols)
          graph.clamp(("eps-gauss", eps_gauss))
        } else if (graph.modelTypeName.contains("piece")) {
          val eps_piece: Mat = rand(n_lat, x.ncols)
          graph.clamp(("eps-piece", eps_piece))
        }

      }
      graph.eval()
      if (graph.modelTypeName.contains("vae")) {
        //If model is variational, use lower-bound to get probs
        val P_theta = sum(graph.getStat("x-out") *@ graph.getStat("x-targ"), 1) //P_\Theta
        var KL_term: Mat = null
        if (graph.modelTypeName.contains("hybrid")) {
          if (null != graph.theta.getParam("gamma")) {
            val KL_gauss = graph.getOp("KL-gauss").per_samp_result
            val KL_piece = graph.getOp("KL-piece").per_samp_result
            KL_term = KL_gauss + KL_piece
            KL_gauss_s = graph.getStat("KL-gauss") //*@ numSamps
            KL_piece_s = graph.getStat("KL-piece") //*@ numSamps
          } else {
            KL_term = 0f
          }
        } else if (graph.modelTypeName.contains("gaussian")) {
          if (null != graph.theta.getParam("gamma")) {
            KL_term = graph.getOp("KL-gauss").per_samp_result
            KL_gauss_s = graph.getStat("KL-gauss") //*@ numSamps
          } else {
            KL_term = 0f
          }
        } else if (graph.modelTypeName.contains("piece")) {
          if (null != graph.theta.getParam("gamma")) {
            KL_term = graph.getOp("KL-piece").per_samp_result
            KL_piece_s = graph.getStat("KL-piece") //*@ numSamps
          } else {
            KL_term = 0f
          }
        }
        vlb = ln(P_theta) - KL_term //variational lower bound log P(X) = (ln P_Theta - KL)
        val full_vlb = exp(ln(graph.getStat("x-out")) - KL_term) //get VLB for ALL WORDS IN EACH PREDICTION
        println("-----")
        println(extractTerms(full_vlb,y,lex))
        println("-----")
      }
      println("\n  ~~> Doc.VLB = "+(sum(vlb)/L_n))
      step += 1
    }
    if(numSGDInfSteps > 1)
      graph.toggleFreezeOp("z",false)

    //Now update our evaluation across data-set w/ found validation lower bound
    if (null != log_probs) {
      log_probs = log_probs + vlb
    } else
      log_probs = vlb //user vlb in place of intractable distribution
    val doc_nll = (sum(log_probs) / L_n) // <-- here we normalize by document lengths

    graph.unfreezeOGraph()
    graph.hardClear()
    graph.muteDerivs(false,graph.theta)
    return (log_probs,doc_nll,KL_gauss_s,KL_piece_s)
  }

  def extractTerms(P_theta : Mat, y : Mat, lex : Lexicon):String = {
    val y_stat = find3(SMat(y))
    val y_argmax = y_stat._1.asInstanceOf[IMat].t
    val y_vals = y_stat._3.asInstanceOf[FMat].t
    val p_argmax = maxi2(P_theta)._2.asInstanceOf[IMat]
    val p_max = maxi(P_theta)
    println("Y.V = " + y_vals)
    println("Y.I = " + y_argmax)
    println("P.V = "+p_argmax)
    println("P.I = "+p_max)
    val y_ind = y_argmax
    val p_ind = p_argmax
    var doc = ""
    var i = 0
    while(i < y_ind.ncols){
      doc += lex.getSymbol(y_ind(0,i)) + " "
      i += 1
    }
    var m_doc = ""
    i = 0
    while(i < p_ind.ncols){
      m_doc += lex.getSymbol(p_ind(0,i)) + " "
      i += 1
    }
    val result = "Y.Doc: "+doc + "\nM.Doc: "+m_doc
    return result
  }

  /**
    *
    * Note that setting numSGDInfSteps > 1 will trigger the more expensive iterative inference
    * proceudre to get latent variables per document.
    *
    * @param rng
    * @param graph
    * @param dataChunks
    * @param numModelSamples -> num samples to compute expectation of lower-bound over for model
    * @param numSGDInfSteps -> default is 1 (non-SGD inference)
    * @param lr_inf -> step-size for SGD iterative inference
    * @return (doc.nll, loss, KL-gaussian-score, KL-piecewise-score)
    */
  def evalModel(rng : Random, graph : OGraph, dataChunks : ArrayList[(Mat,Mat)],
                numModelSamples : Int = 10, numSGDInfSteps : Int = 1, lr_inf : Float = 0.1f, lex : Lexicon = null):Array[Mat] ={
    //Temporarily set any KL max-tricks to 0 to make bound tight...
    val KL_gauss = graph.getOp("KL-gauss")
    var gauss_trick = 0f
    val KL_piece = graph.getOp("KL-piece")
    var piece_trick = 0f
    if(KL_gauss != null){
      gauss_trick = KL_gauss.asInstanceOf[KL_Gauss].maxTrickConstant
      KL_gauss.asInstanceOf[KL_Gauss].maxTrickConstant = 0f
    }
    if(KL_piece != null){
      piece_trick = KL_piece.asInstanceOf[KL_Piece].maxTrickConstant
      KL_piece.asInstanceOf[KL_Piece].maxTrickConstant = 0f
    }

    graph.hardClear()
    val numDocs = dataChunks.size() * 1f
    val stats =new Array[Mat](3)
    var doc_nll:Mat = 0f
    var KL_gauss_score:Mat = 0f
    var KL_piece_score:Mat = 0f
    var numDocsSeen = 0
    var n_lat = graph.getOp("z").dim //we need to know # of latent variables in model
    if(graph.getOp("z-gaussian") != null){
      n_lat = graph.getOp("z-gaussian").dim
    }else if(graph.getOp("z-piece") != null){
      n_lat = graph.getOp("z-piece").dim
    }
    var i = 0
    while(i < dataChunks.size()){
      val batch = dataChunks.get(i)
      val x = batch._1.asInstanceOf[Mat]
      val y = batch._2.asInstanceOf[Mat]

      if(numSGDInfSteps > 1){
        println(" INFER FOR DOC("+i+")")
        val stat = Train.getIterativeBound(rng,graph,x,y,n_lat,numSGDInfSteps,lr_inf,lex)
        val log_probs = stat._1.asInstanceOf[Mat]
        doc_nll += stat._2.asInstanceOf[Mat]
        KL_gauss_score += stat._3.asInstanceOf[Mat]
        KL_piece_score += stat._4.asInstanceOf[Mat]
      }else{
        val stat = Train.getSampledBound(rng,graph,x,y,n_lat,numModelSamples)
        val log_probs = stat._1.asInstanceOf[Mat]
        doc_nll += stat._2.asInstanceOf[Mat]
        KL_gauss_score += stat._3.asInstanceOf[Mat]
        KL_piece_score += stat._4.asInstanceOf[Mat]
      }
      numDocsSeen += 1
      print("\r > "+numDocsSeen + " docs seen...")
      i += 1
    }
    println()
    stats(0) = -doc_nll / (1f * numDocs)
    stats(1) = KL_gauss_score /// (1f * numDocs)
    stats(2) = KL_piece_score  /// (1f * numDocs)

    //Turn back on any KL-max tricks...
    if(KL_gauss != null){
      KL_gauss.asInstanceOf[KL_Gauss].maxTrickConstant = gauss_trick
    }
    if(KL_piece != null){
      KL_piece.asInstanceOf[KL_Piece].maxTrickConstant = piece_trick
    }

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
    val binarizeVectors = configFile.getArg("binarizeVectors").toBoolean
    println(" > Vocab |V| = "+dict.getLexiconSize())
    val trainModel = configFile.getArg("trainModel").toBoolean
    var graphFname = configFile.getArg("graphFname")
    var printDims = false
    if(configFile.isArgDefined("printDims")){
      printDims = configFile.getArg("printDims").toBoolean
    }

    var outputDir:String = null
    if(graphFname.contains("/")){ //extract output directory from graph fname if applicable...
      outputDir = graphFname.substring(0,graphFname.lastIndexOf("/")+1)
      val dir = new File(outputDir)
      dir.mkdir() //<-- build dir on disk if it doesn't already exist...
      graphFname = graphFname.substring(graphFname.indexOf("/")+1)
    }else{
      outputDir = "tmp_model_out/"
      val dir = new File(outputDir)
      dir.delete() //<-- delete dir if it exists
      dir.mkdir() //<-- build dir on disk if it doesn't already exist...
    }

    //Read in config file to get all program meta-parameters
    archBuilder.readConfig(configFile)
    archBuilder.n_in = dict.getLexiconSize() //input/output-dim is = to |V|

    //Build a sampler for main data-set
    val sampler = new DocSampler(dataFname,dict)
    sampler.binarizeVectors = binarizeVectors
    sampler.loadDocsFromLibSVMToCache()

    if(trainModel){ //train/fit model to data
      val numVLBSamps = configFile.getArg("numVLBSamps").toInt
      val miniBatchSize = configFile.getArg("miniBatchSize").toInt
      val numEpochs = configFile.getArg("numEpochs").toInt
      val validFname = configFile.getArg("validFname")
      val errorMark = configFile.getArg("errorMark").toInt
      val norm_rescale = configFile.getArg("norm_rescale").toFloat
      val optimizer = configFile.getArg("optimizer")
      val lr = configFile.getArg("lr").toFloat
      val patience = configFile.getArg("patience").toInt
      val lr_div = configFile.getArg("lr_div").toFloat
      val lr_min = configFile.getArg("lr_min").toFloat
      val moment = configFile.getArg("moment").toFloat
      val epoch_bound = configFile.getArg("epoch_bound").toInt
      val apply_lr_div_per_epoch = configFile.getArg("apply_lr_div_per_epoch").toBoolean
      val train_check_mark = configFile.getArg("train_check_mark").toInt
      if(apply_lr_div_per_epoch){
        println(" > Applying lr-schedule per epoch...")
      }
      val gamma_iter_bound = configFile.getArg("gamma_iter_bound").toInt
      //Build validation set to conduct evaluation
      var validSampler = new DocSampler(validFname,dict)
      validSampler.loadDocsFromLibSVMToCache()
      println(" > Building valid-eval sample...")
      val dataChunks = Train.buildFullSample(validSampler)
      validSampler = null //toss aside this sampler for garbage collection
      var trainChunks:ArrayList[(Mat,Mat)] = null
      if(train_check_mark > 0){
        println(" > Building train-eval sample ("+dataChunks.size() + ")")
        trainChunks = Train.buildRandSample(rng,sampler,dataChunks.size())
      }
      val numSGDInfSteps = configFile.getArg("numSGDInfSteps").toInt
      val lr_inf = configFile.getArg("lr_inf").toFloat

      //Build Ograph given config
      val graph = archBuilder.autoBuildGraph()
      graph.saveOGraph(outputDir+graphFname)
      graph.theta.saveTheta(outputDir+"init")
      var n_lat = graph.getOp("z").dim //we need to know # of latent variables in model
      if(graph.getOp("z-gaussian") != null){
        n_lat = graph.getOp("z-gaussian").dim
      }else if(graph.getOp("z-piece") != null){
        n_lat = graph.getOp("z-piece").dim
      }
      if(printDims){
        println(graph.theta.printDims())
      }
      //Build optimizer given config
      var opt:Opt = null
      val climb_type = "descent"
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
        opt = new SGOpt(lr=lr,moment = moment, opType=climb_type)
      }
      opt.norm_threshold = norm_rescale

      println(" ++++ Model: " + graph.modelTypeName + " Properties ++++ ")
      println("  # Inputs = "+graph.getOp("x-in").dim)
      println("  # Lyr 1 Hiddens = "+graph.getOp("h0").dim)
      if(archBuilder.n_hid_2 > 0){
        println("  # Lyr 2 Hiddens = "+graph.getOp("h1").dim)
      }
      println("  # Latents = "+graph.getOp("z").dim)
      println("  # Outputs = "+graph.getOp("x-out").dim)
      println(" ++++++++++++++++++++++++++ ")

      val logger = new Logger(outputDir + "error_stat.log")
      logger.openLogger()
      logger.writeStringln("Epoch, V.NLL, V.PPL, V.KL-G, V.KL-P,Avg.Up-T,Max.Up-T,T.NLL,T.PPL,T.KL_G,T.KL-P")

      var stats = Train.evalModel(rng,graph,dataChunks,numModelSamples = numVLBSamps,
        numSGDInfSteps = numSGDInfSteps, lr_inf = lr_inf)
      var bestNLL = stats(0)
      var bestPPL = exp(bestNLL)
      println("-1 > NLL = "+bestNLL + " PPL = " + bestPPL + " KL.G = "+stats(1) + " KL.P = "+stats(2))

      if(train_check_mark > 0) {
        stats = Train.evalModel(rng, graph, trainChunks, numModelSamples = numVLBSamps,
          numSGDInfSteps = numSGDInfSteps, lr_inf = lr_inf)
        val trainNLL = stats(0)
        val trainPPL = exp(trainNLL)
        val trainGKL = stats(1)
        val trainPKL = stats(2)
        println("     Train.NLL = " + trainNLL + " PPL = " + trainPPL + " KL.G = " + trainGKL + " KL.P = " + trainPKL)
        logger.writeStringln("-1"+","+bestNLL+","+bestPPL+","+stats(1)+","+stats(2)+",NA,NA," + trainNLL
          + "," + trainPPL + "," + trainGKL + "," + trainPKL)
      }else
        logger.writeStringln("-1"+","+bestNLL+","+bestPPL+","+stats(1)+","+stats(2)+",NA,NA,0,0,0,0")

      //Actualy train model
      var totalNumIter = 0
      val gamma_delta = (1f - archBuilder.vae_gamma)/(gamma_iter_bound*1f)
      var impatience = 0
      var epoch = 0
      var worst_case_update_time = 0f
      //var worst_case_prep_time = 0f
      //var worst_case_grad_time = 0f
      var numIter = 0
      while(epoch < numEpochs) {
        if(epoch == (numEpochs-1)){
          opt.setPolyakAverage()
        }
        var numSampsSeen = 0 // # samples seen w/in an epoch
        var mark = 1
        var currNLL:Mat = bestNLL
        var currPPL:Mat = bestPPL
        var avg_update_time = 0f
        var numEpochIter = 0f // num iterations w/in an epoch (i.e., until sampler depleted fully)
        while (sampler.isDepleted() == false) {
          var t0 = System.nanoTime()
          /* ####################################################
           * Gather & clamp data/samples to OGraph
           * ####################################################
           */
          val batch = sampler.drawMiniBatch(miniBatchSize, rng)
          val x = batch._1.asInstanceOf[Mat]
          val y = batch._2.asInstanceOf[Mat]
          //var t1 = System.nanoTime()
          //worst_case_prep_time = Math.max(worst_case_prep_time,(t1 - t0))

          val numSamps = y.ncols
          numSampsSeen += numSamps
          numIter += 1
          totalNumIter += 1
          graph.hardClear()
          graph.clamp(("x-in",x))
          graph.clamp(("x-targ",y))
          graph.clamp(("N",1f * numSamps)) //<-- note: mean loss is more stable for learning

          //Generate random samples for model is needed
          if(graph.modelTypeName.contains("hybrid")){
            val eps_gauss:Mat = normrnd(0f,1f,n_lat,x.ncols)
            val eps_piece:Mat = rand(n_lat,x.ncols)
            graph.clamp(("eps-gauss",eps_gauss))
            graph.clamp(("eps-piece",eps_piece))
          }else if(graph.modelTypeName.contains("gaussian")){
            val eps_gauss:Mat = normrnd(0f,1f,n_lat,x.ncols)
            graph.clamp(("eps-gauss",eps_gauss))
          }else if(graph.modelTypeName.contains("piece")){
            val eps_piece:Mat = rand(n_lat,x.ncols)
            graph.clamp(("eps-piece",eps_piece))
          }

          //t0 = System.nanoTime()
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
          //t1 = System.nanoTime()
          //worst_case_grad_time = Math.max(worst_case_grad_time,(t1 - t0))

          //t0 = System.nanoTime()
          /* ####################################################
           * Update model given gradients
           * ####################################################
           */
          opt.update(theta = graph.theta, nabla = grad, miniBatchSize = numSamps)

          if(gamma_iter_bound > 0){
            val gamma = Math.min(1f,graph.theta.getParam("gamma").dv.toFloat + gamma_delta)
            graph.theta.setParam("gamma",gamma)
          }

          var t1 = System.nanoTime()
          worst_case_update_time = Math.max(worst_case_update_time,(t1 - t0))
          avg_update_time += (t1 - t0)
          numEpochIter += 1

          if(sampler.isDepleted() || (errorMark > 0 && numSampsSeen >= (mark * errorMark))){ //eval model @ this point
            stats = Train.evalModel(rng,graph,dataChunks,numModelSamples = numVLBSamps,
              numSGDInfSteps = numSGDInfSteps, lr_inf = lr_inf)
            currNLL = stats(0)
            //logger.writeStringln("Epoch, Valid.NLL, Valid.PPL, KL-Gauss, KL-Piece")
            currPPL = exp(currNLL)
            if(currNLL.dv.toFloat <= bestNLL.dv.toFloat){
              bestNLL = currNLL
              bestPPL = currPPL
              graph.theta.saveTheta(outputDir+"best_at_epoch_"+epoch)
              impatience = Math.max(0,impatience - 1)
            }else{
              if(apply_lr_div_per_epoch){
                if(sampler.isDepleted()){
                  if (impatience >= patience) {
                    opt.stepSize = Math.max(lr_min, opt.stepSize / lr_div)
                    println("\n  Annealed LR to "+opt.stepSize)
                    impatience = 0
                  }else{
                    impatience += 1
                  }
                }
              }else {
                if (lr_div > 0 && epoch_bound > 0 && lr_min > 0 && patience > 0) {
                  if (epoch >= epoch_bound) {
                    if (impatience >= patience) {
                      //adjust learning rate, reset impatience
                      opt.stepSize = Math.max(lr_min, opt.stepSize / lr_div) // keeps lr from going below some bound
                      impatience = 0
                    } else //increment impatience
                      impatience += 1
                  } //else haven't reach appropriate epoch to start annealing just yet...
                } //else ignore annealing schedule, conditions are invalid
              }
            }
            if(sampler.isDepleted()){
              graph.theta.saveTheta(outputDir+"check_epoch_"+epoch)
            }
            mark += 1
            println("\n "+epoch+" >> NLL = "+currNLL + " PPL = " + currPPL + " KL.G = "+stats(1) + " KL.P = "+stats(2) + " over "+numSampsSeen + " samples")
            if(train_check_mark > 0){
              if(sampler.isDepleted() && (epoch + 1) % train_check_mark == 0){
                stats = Train.evalModel(rng,graph,trainChunks,numModelSamples = numVLBSamps,
                  numSGDInfSteps = numSGDInfSteps, lr_inf = lr_inf)
                val trainNLL = stats(0)
                val trainPPL = exp(trainNLL)
                val trainGKL = stats(1)
                val trainPKL = stats(2)
                println("   >> Train.NLL = " + trainNLL +" PPL = "+trainPPL+" KL.G = "+trainGKL+ " KL.P = "+trainPKL)
                logger.writeStringln("" + epoch + "," + currNLL + "," + currPPL + "," + stats(1) + "," + stats(2) + ","
                  + (avg_update_time / numEpochIter * 1e-9f) + "," + (worst_case_update_time * 1e-9f)+","+trainNLL
                  +","+trainPPL+","+trainGKL+","+trainPKL)
              }else{
                logger.writeStringln("" + epoch + "," + currNLL + "," + currPPL + "," + stats(1) + "," + stats(2) + ","
                  + (avg_update_time / numEpochIter * 1e-9f) + "," + (worst_case_update_time * 1e-9f)+",0,0,0,0")
              }
            }else {
              logger.writeStringln("" + epoch + "," + currNLL + "," + currPPL + "," + stats(1) + "," + stats(2) + ","
                + (avg_update_time / numEpochIter * 1e-9f) + "," + (worst_case_update_time * 1e-9f)+",0,0,0,0")
            }
          }
          print("\r " +" S.len = " +sampler.ptrs.size() + " D.len = " + sampler.depleted_ptrs.size()
            +" T.avg = "+ (avg_update_time/numEpochIter * 1e-9f) + " T.worst = " + ((worst_case_update_time * 1e-9f))
            + " s, over "+numSampsSeen + " samples"
          )
        }
        println("\n---------------")
        var polyak_avg:Theta = null
        if(epoch == (numEpochs-1)){
          println(" >> Estimating Polyak average over Theta...")
          polyak_avg = opt.estimatePolyakAverage()
          polyak_avg.saveTheta(outputDir+"polyak_avg")
        }
        epoch += 1
        sampler.reset()
      }
      println(" Best.Valid.NLL = "+bestNLL + " Valid.PPL = "+bestPPL)
    }else{ //evaluation only
      val numEvalSamps = configFile.getArg("numEvalSamps").toInt // < 0 turns this off
      val numVLBSamps = configFile.getArg("numVLBSamps").toInt
      val numTrials = configFile.getArg("numTrials").toInt // < 0 turns this off
      val thetaFname = configFile.getArg("thetaFname")
      val numSGDInfSteps = configFile.getArg("numSGDInfSteps").toInt
      val lr_inf = configFile.getArg("lr_inf").toFloat
      println(" > Loading Theta: "+thetaFname)
      val theta = archBuilder.loadTheta(thetaFname)
      //Load graph given config
      println(" > Loading OGraph: "+outputDir+graphFname)
      val graph = archBuilder.loadOGraph(outputDir+graphFname)
      graph.theta = theta
      graph.hardClear() //<-- clear out any gunked up data from previous sessions
      println(" > Building data-set...")
      if(numEvalSamps > 0 && numTrials > 0){ //sampled evaluation ala old-school Hinton-style =)
        var mean_nll:Mat = 0f
        var mean_ppl:Mat = 0f
        var stdDev_nll:Mat = 0f
        var stdDev_ppl:Mat = 0f
        val nll_s = new Array[Mat](numTrials)
        val ppl_s = new Array[Mat](numTrials)
        var trial = 0
        while(trial < numTrials){
          val dataChunks = Train.buildRandSample(rng,sampler,numEvalSamps)
          val stats = Train.evalModel(rng, graph, dataChunks, numModelSamples = numVLBSamps,
            numSGDInfSteps = numSGDInfSteps, lr_inf = lr_inf,dict)
          val nll = stats(0)
          val ppl = exp(nll)
          nll_s(trial) = nll
          mean_nll += nll
          ppl_s(trial) = ppl
          mean_ppl += ppl
          println(" >> Trail "+trial + " NLL = "+nll + " PPL = "+ppl)
          trial += 1
          sampler.reset()
        }
        // Now calculate statistics: mean & std devs
        mean_nll = (mean_nll/(1f * numTrials))
        mean_ppl = (mean_ppl/(1f * numTrials))
        var i = 0
        while(i < nll_s.length){
          stdDev_nll += (nll_s(i) - mean_nll) *@ (nll_s(i) - mean_nll)
          stdDev_ppl += (ppl_s(i) - mean_ppl) *@ (ppl_s(i) - mean_ppl)
          i += 1
        }
        stdDev_nll = sqrt(stdDev_nll/((1f * numTrials) - 1f))
        stdDev_ppl = sqrt(stdDev_ppl/((1f * numTrials) - 1f))
        println(" ====== Performance Statistics ======")
        println(" > Avg.NLL = " + mean_nll + " +/- " + stdDev_nll)
        println(" > Avg.PPL = " + mean_ppl + " +/- " + stdDev_ppl)
      }else {
        val dataChunks = Train.buildFullSample(sampler)
        println(" > Evaluating model on data-set...")
        //graph.muteEvals(true,"L") //avoid calculating outermost-loss
        val stats = Train.evalModel(rng, graph, dataChunks, numModelSamples = numVLBSamps,
          numSGDInfSteps = numSGDInfSteps, lr_inf = lr_inf,dict)
        val nll = stats(0)
        val ppl = exp(nll)
        println(" ====== Performance ======")
        println(" > Corpus.NLL = " + nll)
        println(" > Corpus.PPL = " + ppl)
        println(" > over " + dataChunks.size() + " documents")
      }
    }
  }


}
