import java.util.{ArrayList, HashMap}

import BIDMat._
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import java.util.Random

import YADLL.Utils.MiscUtils
import YAVL.Data.Text.Doc
import YAVL.Data.Text.Lexicon.Lexicon
import YAVL.DataStreams.Text.DocStream
import YAVL.Utils.ScalaDebugUtils

/**
  * Transforms a corpus into an array of DocSamples to be sampled (w/o replacement) from.
  *
  * Created by ago109 on 10/27/16.
  */
class DocSampler(var fname : String, var dict : Lexicon, var cacheSize : Int = 100, var tokenType : String = "word") {
  //val numLines = io.Source.fromFile(fname).getLines.size
  //Stores in RAM only so many doc-samples to prevent a memory-driven slow-down
  var cache = new ArrayList[DocSample]()
  var depletedSamps = new ArrayList[DocSample]()
  var dim = this.dict.getLexiconSize()
  //this.loadDocsFromStreamToCache()

  def numDocs():Int={
    return cache.size()
  }

  def loadDocsFromTSVToCache():Unit = {//assumes no header...
    val fd = MiscUtils.loadInFile(this.fname)
    var line = fd.readLine()
    while(line != null){
      var tok = line.replaceAll(" ","").split("\t")
      val doc_id = tok(0).toInt
      tok = tok(1).split(",")
      val idx_val_map = new HashMap[Integer,java.lang.Float]()
      var i = 0
      while(i < tok.length){
        val sub_tok = tok(i).split(":")
        val idx = sub_tok(0).toInt
        val value = sub_tok(1).toFloat
        var currVal = idx_val_map.get(idx)
        if(currVal != null){
          currVal += 1
          idx_val_map.put(idx,currVal)
        }else{
          idx_val_map.put(idx,value)
        }
        i += 1
      }
      val sample = new DocSample(doc_id,this.dim)
      sample.buildFrom(idx_val_map, applyTransform = false) ///DO NOT re-apply log-transform
      this.cache.add(sample)
      print("\r " + this.cache.size() + " docs converted to bag-of-words...")
      line = fd.readLine()
    }
    fd.close()
    println()
  }


  def loadDocsFromStreamToCache(): Unit ={
    val tmpStream = new DocStream(fname)
    tmpStream.setTokenType(this.tokenType)
    tmpStream.lowerCaseDocs(false)
    var idx = 0
    var doc: Doc = tmpStream.nextDoc()
    while (!tmpStream.atEndOfStream() || doc != null) {
      val idx_val_map = new HashMap[Integer, java.lang.Float]()
      while (doc.atEOF() == false) {
        val symb = doc.getCurrentSymbol()
        val idx = dict.getIndex(symb)
        if (idx != null) {
          var currVal = idx_val_map.get(idx)
          if (currVal != null) {
            currVal += 1f
            idx_val_map.put(idx, currVal)
          } else {
            idx_val_map.put(idx, 1f)
          }
        }
        doc.advanceSymbolPtr()
      }
      val sample = new DocSample(idx,dict.getLexiconSize())
      this.cache.add(sample)
      print("\r " + this.cache.size() + " docs converted to bag-of-words...")
      idx += 1
      doc = tmpStream.nextDoc() //grab next doc from Doc-Stream
    }
    println()
  }

  def isDepleted(): Boolean ={
    if(this.cache.size() == 0){
      return true
    }
    return false
  }

  def reset(): Unit ={
    this.cache = this.depletedSamps
    this.depletedSamps = new ArrayList[DocSample]()
    var i = 0
    while(i < this.cache.size()){
      this.cache.get(i).resetTargetIdx()
      i += 1
    }
  }

  private def drawRandDocSample(rng : Random):(IMat,FMat,Int) ={
    val ptr = MiscUtils.genRandInt(rng,0,this.cache.size())
    val doc = this.cache.get(ptr)
    val idx = new IMat(doc.bagOfIdx.length,1,doc.bagOfIdx)
    val vals = new FMat(doc.bagOfVals.length,1,doc.bagOfVals)
    val target = doc.drawTargetIdx()
    if(doc.isDepleted()){
      val doc = this.cache.remove(ptr) //remove depleted doc from sample-cache
      this.depletedSamps.add(doc)
    }
    return (idx,vals,target)
  }

  def drawMiniBatch(rng : Random, n : Int):(Mat,Mat) ={
    if(this.cache.size() > 0){
      var x_col:Mat = null
      var x_ind:Mat = null
      var x_val:Mat = null
      var y_col:Mat = null
      var y_ind:Mat = null
      var y_val:Mat = null
      var mb_size = 0
      var col_ptr = 0
      while(this.cache.size() > 0 && mb_size < n) {
        val samp = this.drawRandDocSample(rng)
        val x_i = samp._1.asInstanceOf[IMat]
        val x_v = samp._2.asInstanceOf[FMat]
        val y_i = samp._3.asInstanceOf[Int]
        val y_v = 1f // y_i value is always a 1-hot encoding
        if(null != x_ind){
          x_ind = x_ind on x_i
          x_val = x_val on x_v
          x_col = x_col on iones(x_i.nrows,1) *@ col_ptr
          y_ind = y_ind on y_i
          y_val = y_val on y_v
          y_col = y_col on col_ptr
        }else{
          x_ind = x_i
          x_val = x_v
          x_col = iones(x_i.nrows,1) *@ col_ptr
          y_ind = y_i
          y_val = y_v
          y_col = col_ptr
        }
        col_ptr += 1
        mb_size += 1
      }
      //println("x.ind: "+x_ind.t)
      //println("y.ind: "+y_ind.t)
      //Now compose the mini-batch (x,y) with whatever we could scrape
      val x = sparse(IMat(x_ind),IMat(x_col),FMat(x_val),this.dim,mb_size)
      val y = sparse(IMat(y_ind),IMat(y_col),FMat(y_val),this.dim,mb_size)
      return (x,y)
    }
    System.err.println(" ERROR: Cannot draw from an empty sample-cache....")
    return null
  }

  //Code to start a dynamic sample-cache policy
  /*
  var ptrs = new ArrayList[Int]()
  var depleted_ptrs = new ArrayList[Int]()

  private def initPtrs(): Unit ={
    var i = 0
    while(i < numLines){
      ptrs.add(i)
      i += 1
    }
  }

  def reset(): Unit ={
    this.ptrs = this.depleted_ptrs
    this.depleted_ptrs = new ArrayList[Int]()
  }


  def manageCache(rng : Random ): Unit ={ //watch-dog function
    if(this.ptrs.size() > 0 && this.cache.size() < this.cacheSize){
      //Replenish cache if possible
      val ptrs_to_read = new HashMap[Integer,Integer]()
      while(this.cache.size() < this.cacheSize && this.ptrs.size() > 0){
        val ptr = MiscUtils.genRandInt(rng,0,this.ptrs.size())
        val samp_ptr = this.ptrs.remove(ptr)
        ptrs_to_read.put(samp_ptr,samp_ptr)
        this.depleted_ptrs.add(samp_ptr)
      }
      //Now read the drawn pointers from the file on disk...


    }else{
      System.err.println(" ERROR: Stream of samples has been depleted!!")
    }
  }
  */


}
