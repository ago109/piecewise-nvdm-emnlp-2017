import BIDMat.SciFunctions._
import YADLL.FunctionGraph.{BuildArch, Theta}
import YADLL.Utils.{ConfigFile, MathUtils}
import YAVL.Data.Text.Lexicon.Lexicon
import YAVL.Utils.Logger

import scala.runtime.RichInt
//Imports from BIDMat
import BIDMat.{CMat, CSMat, DMat, Dict, FMat, GIMat, GMat, GSMat, HMat, IDict, IMat, Mat, SBMat, SDMat, SMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._

/**
  * Builds a set of "document vectors" for an input corpus under the given model. (these latent variable
  * vectors are to be used in visualization tasks via t-SNE, the Barnes Hutte approximation).
  * Note that output vectors are TSV files to play nicely with Maaten's Barnes Hutte t-SNE code.
  *
  * Created by ago109 on 11/14/16.
  */
object DocVectBuilder {
  Mat.checkMKL //<-- needed in order to check for BIDMat on cpu or gpu...

  def main(args : Array[String]): Unit ={
    if(args.length != 1){
      System.err.println("usage: [/path/to/config.cfg]")
      return
    }
    //Extract arguments
    val configFile = new ConfigFile(args(0)) //grab configuration from local disk
    val dataFname = configFile.getArg("dataFname")
    val dictFname = configFile.getArg("dictFname")
    val graphFname = configFile.getArg("graphFname")
    val thetaFname = configFile.getArg("thetaFname")
    val outputFname = configFile.getArg("outputFname")
    val outputLabFname = configFile.getArg("outputLabFname")
    val lr_gauss = configFile.getArg("lr_gauss").toFloat
    val lr_piece = configFile.getArg("lr_piece").toFloat
    val gauss_norm = configFile.getArg("gauss_norm").toFloat
    val piece_norm = configFile.getArg("piece_norm").toFloat
    val numSGDInfSteps = configFile.getArg("numSGDInfSteps").toInt
    val patience = configFile.getArg("patience").toInt

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

    val fd_dat = new Logger(outputFname)
    fd_dat.openLogger()
    val fd_lab = new Logger(outputLabFname)
    fd_lab.openLogger()

    var numDocSeen = 0
    //Begin potential iterative inference procedure to find optimal latents per document
    while(sampler.isDepleted() == false) {
      val doc = sampler.drawFullDocBatch()
      val x = doc._1.asInstanceOf[Mat]
      val y = doc._2.asInstanceOf[Mat]
      val id = doc._3.asInstanceOf[Int]

      graph.muteDerivs(true,graph.theta) //fix \Theta (no cheating! ;) )
      //val L_n = x.ncols * 1f //get per-document lengths
      var KL_gauss_s:Mat = 0f
      var KL_piece_s:Mat = 0f
      //Evaluate model on x using y as the target
      graph.clamp(("x-in", x)) //<-- BOW, document
      graph.clamp(("x-targ", y)) //<-- target word(s) in document

      //Generate statistics for model
      //One latent variable copied N times
      var eps_gauss: Mat = ones(n_lat,x.ncols) *@ normrnd(0f, 1f, n_lat, 1)
      var eps_piece: Mat = ones(n_lat,x.ncols) *@ rand(n_lat, 1)
      val numDocs = 1 //<-- all of the predictions are for a single document
      val KL_correction:Mat = ones(1,x.ncols) *@ x.ncols //correct for KL being copied equal to # of predictions

      graph.clamp(("eps-gauss", eps_gauss))
      graph.clamp(("eps-piece", eps_piece))
      graph.clamp(("KL-correction",KL_correction))
      graph.clamp(("N",numDocs)) //<-- note we don't want a mean loss this time...
      //graph.clamp(("logit-wghts",x)) // <-- original BOW serves as weights to model's logits

      // Infer/estimate posterior probabilities
      var best_doc_nll = 10000f
      var impatience = 0
      var vlb:Mat = null
      var best_vlb:Mat = null
      var optimal_Z:Mat = null
      var step = 0
      while(step < numSGDInfSteps && impatience < patience) {
        if(step > 0){ //for every step beyond the first, we perform an iterative inference step...
          //We have to carefully freeze the right parts of the graph to do proper partial SGD-inference
          if (graph.modelTypeName.contains("hybrid") || graph.modelTypeName.contains("gaussian")) {
            graph.toggleFreezeOp("mu-prior", true)
            graph.toggleFreezeOp("sigma-prior", true)
            graph.toggleFreezeOp("mu", true)
            graph.getOp("mu").muteEval(false) //to allow a gradient to be computed w/ respect to this
            graph.toggleFreezeOp("sigma", true)
            graph.getOp("sigma").muteEval(false) //to allow a gradient to be computed w/ respect to this
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
            val mu_post_rho = MathUtils.clip_by_norm(grad.getParam("mu"),gauss_norm)
            val sigma_post_rho = MathUtils.clip_by_norm(grad.getParam("sigma"),gauss_norm)
            val mu_post = graph.getStat("mu")
            val sigma_post = graph.getStat("sigma")
            graph.getOp("mu").result = mu_post - (mu_post_rho *@ lr_gauss)
            graph.getOp("sigma").result = sigma_post - (sigma_post_rho *@ lr_gauss)
            graph.getOp("mu").muteEval(true)
            graph.getOp("sigma").muteEval(true)
          }
          if (graph.modelTypeName.contains("hybrid") || graph.modelTypeName.contains("piece")) {
            val a_post_name = "a-post-"
            var i = 0
            while(graph.getOp((a_post_name + i)) != null){
              val a_post_rho = MathUtils.clip_by_norm(grad.getParam(a_post_name + i),piece_norm)
              val a_post = graph.getStat(a_post_name+i)
              //println("a-post-"+i+".before:\n"+a_post)
              //println("a_post_rho-"+i+":\n"+a_post_rho)
              graph.getOp(a_post_name+i).result = a_post - (a_post_rho *@ (lr_piece))
              graph.getOp(a_post_name+i).muteEval(true)
              //println("a-post-"+i+".after:\n"+graph.getStat(a_post_name+i))
              i += 1
            }
          }
          //Now that posterior parameters have been update, we generate fresh samples to re-compute z
          eps_gauss = ones(n_lat,x.ncols) *@ normrnd(0f, 1f, n_lat, 1)
          eps_piece = ones(n_lat,x.ncols) *@ rand(n_lat, 1)
          graph.clamp(("eps-gauss", eps_gauss))
          graph.clamp(("eps-piece", eps_piece))
          graph.clamp(("KL-correction",KL_correction))
          graph.clamp(("N",numDocs)) //<-- note we don't want a mean loss this time...
          graph.clamp(("logit-wghts",x)) // <-- original BOW serves as weights to model's logits
        } //else, do nothing...
        graph.eval()
        if (graph.modelTypeName.contains("vae")) {
          //If model is variational, use lower-bound to get probs
          //val P_theta = sum(graph.getStat("x-out") *@ graph.getStat("x-targ"), 1) //P_\Theta
          var KL_term: Mat = null
          if (graph.modelTypeName.contains("hybrid")) {
            val KL_gauss = graph.getOp("KL-gauss").per_samp_result
            val KL_piece = graph.getOp("KL-piece").per_samp_result
            KL_term = KL_gauss + KL_piece
            KL_gauss_s = graph.getStat("KL-gauss")
            KL_piece_s = graph.getStat("KL-piece")
          } else if (graph.modelTypeName.contains("gaussian")) {
            KL_term = graph.getOp("KL-gauss").per_samp_result
            KL_gauss_s = graph.getStat("KL-gauss")
          } else if (graph.modelTypeName.contains("piece")) {
            KL_term = graph.getOp("KL-piece").per_samp_result
            KL_piece_s = graph.getStat("KL-piece")
          }
          //vlb = ln(P_theta) - KL_term //variational lower bound log P(X) = (ln P_Theta - KL)
          //val nll = (-(sum(sum(ln( graph.getStat("x-out") + 1e-8f) *@ graph.getStat("x-targ"))) - sum(sum(KL_term))))
          //println("  NLL(Doc) = " + nll) //exp of log of sums = exp of log of products
          vlb = -graph.getStat("L") //VLB is simply negative of loss function
          vlb = sum(vlb,1) //get individual logits for this doc
        }else{
          System.err.println(" ERROR: Model is not some form of VAE? "+graph.modelTypeName)
        }
        val doc_nll = (-sum(sum(vlb))/numDocs).dv.toFloat //estimate doc NLL
        if(doc_nll > best_doc_nll){
          impatience += 1
        }else {
          impatience = Math.max(0,impatience-1)
          best_doc_nll = doc_nll
          best_vlb = vlb
          optimal_Z = graph.getStat("z")
        }
        //println("\n  ~~> Doc.NLL = "+(-sum(sum(vlb))/numDocs) + " G-KL = "+KL_gauss_s + " P-KL = "+KL_piece_s )
        step += 1
      }
      if(null == optimal_Z){
        optimal_Z = graph.getStat("z")
      }
      graph.unfreezeOGraph() //clears all the partial freezing done in this routine
      graph.muteDerivs(false,graph.theta) //un-fixes \Theta
      graph.hardClear() //clears away gunked-up statistcs

      //We may now extract the near-optimal latent & write to disk
      val Z = optimal_Z(?,0) //we only want a single copy (not the mini-batch replication)
      //Store latent variable to disk
      val Z_str = ("" + Z.t).replaceAll(",","\t") //transpose to column-major and convert to string in TSV form
      fd_dat.writeStringln(Z_str)
      //Store mapped label to disk as well
      fd_lab.writeStringln(""+id)
      numDocSeen += 1
      print("\r > "+numDocSeen + " doc-vectors inferred...")
    }
    println()
    fd_dat.closeLogger()
    fd_lab.closeLogger()

  }


}
