import java.util.{ArrayList, HashMap}

import BIDMat._
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import java.util.Random

import YADLL.Utils.MiscUtils
import YAVL.TextStream.Dict.Lexicon
import YAVL.Utils.ScalaDebugUtils

/**
  * Transforms a corpus into an array of DocSamples to be sampled (w/o replacement) from.
  *
  * Created by ago109 on 10/27/16.
  */
class DocSampler(var fname : String, var dict : Lexicon, var cacheSize : Int = 100, var tokenType : String = "word") {
  //val numLines = io.Source.fromFile(fname).getLines.size
  //Stores in RAM only so many doc-samples to prevent a memory-driven slow-down
  var binarizeVectors = false
  var cache = new ArrayList[DocSample]()
  var totalNumDocs = -1
  var dim = this.dict.getLexiconSize()
  //this.loadDocsFromStreamToCache()

  var ptrs = new ArrayList[Int]()
  var depleted_ptrs = new ArrayList[Int]()

  def numDocs():Int={
    return this.totalNumDocs
  }

  /**
    * Reads in a libsvm format file (as a result, this method will convert 1-based indexing to 0-based indexing).
    */
  def loadDocsFromLibSVMToCache():Unit = {//assumes no header...
    val fd = MiscUtils.loadInFile(this.fname)
    if(this.binarizeVectors){
      println(" >> Binarizing document vectors...")
    }
    var maxNonZeros = 0
    var numDiscards = 0
    var line = fd.readLine()
    var s_idx = 0
    while(line != null){
      val tok = line.split("\\s+") //split doc into pieces
      val doc_id = tok(0).toInt //1st item is always document index
      //rest of n-1 items are index:value pairs to be parsed
      val idx_val_map = new HashMap[Integer,java.lang.Float]()
      var i = 1
      while(i < tok.length){
        val sub_tok = tok(i).split(":")
        val idx = sub_tok(0).toInt - 1 // converts 1-based index to 0-based index
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
      if(idx_val_map.size() > 0) {
        val sample = new DocSample(doc_id, this.dim)
        sample.buildFrom(idx_val_map, applyTransform = false, binarizeVectors = this.binarizeVectors) ///DO NOT re-apply log-transform
        maxNonZeros = Math.max(maxNonZeros, sample.bagOfVals.length)
        this.cache.add(sample)
        this.ptrs.add(s_idx)
        s_idx += 1
        print("\r " + this.ptrs.size() + " docs converted to bag-of-words...")
      }else
        numDiscards += 1
      line = fd.readLine()
    }
    fd.close()
    println()
    println(" > Worst-case density/sparsity = "+maxNonZeros + " / "+this.dim)
    println(" > Discared "+numDiscards + " empty docs...")
    this.totalNumDocs = this.cache.size()
  }

  /*
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
    this.totalNumDocs = this.cache.size()
  }
  */

  def isDepleted(): Boolean ={
    if(this.ptrs.size() == 0){
      return true
    }
    return false
  }

  def reset(): Unit ={
    if(this.ptrs.size() == 0)
      this.ptrs = this.depleted_ptrs
    else{ // a simple pre-caution to make sure un-sampled ptrs are NOT overriden...
      this.ptrs.addAll( this.depleted_ptrs )
    }
    this.depleted_ptrs = new ArrayList[Int]()
    //this.cache = this.depletedSamps
    //this.depletedSamps = new ArrayList[DocSample]()
    var i = 0
    while(i < this.cache.size()){
      this.cache.get(i).resetTargetIdx()
      i += 1
    }
  }

  def drawFullRandDoc(rng : Random):(Mat,Mat) ={ //draws a full doc mini-batch
    var idx = MiscUtils.genRandInt(rng,0,this.ptrs.size())
    idx = this.ptrs.remove(idx)
    val doc = this.cache.get(idx)
    this.depleted_ptrs.add(idx)
    //Get bag of words representation
    val x_i = new IMat(doc.bagOfIdx.length,1,doc.bagOfIdx)
    val x_v = new FMat(doc.bagOfVals.length,1,doc.bagOfVals)
    var x_col:Mat = null
    var x_ind:Mat = null
    var x_val:Mat = null
    var y_col:Mat = null
    var y_ind:Mat = null
    var y_val:Mat = null
    var mb_size = 0
    var col_ptr = 0
    while(doc.isDepleted() == false){
      val y_i = doc.drawTargetIdx(rng)
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
    //Now compose the mini-batch (x,y) with whatever we could scrape
    val x = sparse(IMat(x_ind),IMat(x_col),FMat(x_val),this.dim,mb_size)
    val y = sparse(IMat(y_ind),IMat(y_col),FMat(y_val),this.dim,mb_size)
    return (x,y)
  }

  private def drawDocSample(rng : Random = null):(IMat,FMat,Int,Int) ={
    var ptr_idx = 0
    if(rng != null)
      ptr_idx = MiscUtils.genRandInt(rng,0,this.ptrs.size())
    val doc = this.cache.get(this.ptrs.get(ptr_idx))
    val doc_id = doc.doc_id
    val idx = new IMat(doc.bagOfIdx.length,1,doc.bagOfIdx)
    val vals = new FMat(doc.bagOfVals.length,1,doc.bagOfVals)
    val target = doc.drawTargetIdx(rng)
    if(doc.isDepleted()){
      val ptr = this.ptrs.remove(ptr_idx)
      this.depleted_ptrs.add(ptr)
      //val doc = this.cache.remove(ptr_idx) //remove depleted doc from sample-cache
      //this.depletedSamps.add(doc)
    }
    return (idx,vals,target,doc_id)
  }

  def getDocBOWAt(ptr_idx : Int):Mat={
    val doc = this.cache.get(this.ptrs.get(ptr_idx))
    val bow = doc.getBOWVec()
    return bow
  }

  def getDocAt(ptr_idx : Int):DocSample={
    return this.cache.get(this.ptrs.get(ptr_idx))
  }

  /**
    *
    * @return (x,y,doc_id/lab_id)
    */
  def drawFullDocBatch():(Mat,Mat,Int) ={
    if(this.ptrs.size() > 0){
      var x_col:Mat = null
      var x_ind:Mat = null
      var x_val:Mat = null
      var y_col:Mat = null
      var y_ind:Mat = null
      var y_val:Mat = null
      var mb_size = 0
      var col_ptr = 0
      var cache_delta = 0
      val orig_cache_size = this.ptrs.size()
      var doc_id = -1
      while(this.ptrs.size() > 0 && cache_delta == 0) {
        val samp = this.drawDocSample()
        val x_i = samp._1.asInstanceOf[IMat]
        val x_v = samp._2.asInstanceOf[FMat]
        val y_i = samp._3.asInstanceOf[Int]
        if(doc_id < 0){
          doc_id = samp._4.asInstanceOf[Int]
        }
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
        cache_delta = orig_cache_size - this.ptrs.size()
      }
      /*
      println(x_col.t)
      println("x.ind: "+x_ind.t)
      println(y_col.t)
      println("y.ind: "+y_ind.t)
      */
      //Now compose the mini-batch (x,y) with whatever we could scrape
      val x = sparse(IMat(x_ind),IMat(x_col),FMat(x_val),this.dim,mb_size)
      val y = sparse(IMat(y_ind),IMat(y_col),FMat(y_val),this.dim,mb_size)
      return (x,y,doc_id)
    }
    System.err.println(" ERROR: Cannot draw from an empty sample-cache....")
    return null
  }

  /**
    * Draws a mini-batch containing random input-output pairs across all documents in training set.
    * @param n
    * @param rng
    * @return (x.mb, y.mb, doc_ID.mb)
    */
  def drawMiniBatch(n: Int, rng: Random):(Mat,Mat,IMat) ={
    if(this.ptrs.size() > 0){
      var x_col:Mat = null
      var x_ind:Mat = null
      var x_val:Mat = null
      var y_col:Mat = null
      var y_ind:Mat = null
      var y_val:Mat = null
      var docID:IMat = null
      var mb_size = 0
      var col_ptr = 0
      while(this.ptrs.size() > 0 && mb_size < n) {
        val samp = this.drawDocSample(rng)
        val x_i = samp._1.asInstanceOf[IMat]
        val x_v = samp._2.asInstanceOf[FMat]
        val y_i = samp._3.asInstanceOf[Int]
        val doc_id = samp._4.asInstanceOf[Int]
        val y_v = 1f // y_i value is always a 1-hot encoding
        if(null != x_ind){
          x_ind = x_ind on x_i
          x_val = x_val on x_v
          x_col = x_col on iones(x_i.nrows,1) *@ col_ptr
          y_ind = y_ind on y_i
          y_val = y_val on y_v
          y_col = y_col on col_ptr
          docID = docID \ doc_id
        }else{
          x_ind = x_i
          x_val = x_v
          x_col = iones(x_i.nrows,1) *@ col_ptr
          y_ind = y_i
          y_val = y_v
          y_col = col_ptr
          docID = doc_id
        }
        col_ptr += 1
        mb_size += 1
      }
      /*
      println(x_col.t)
      println("x.ind: "+x_ind.t)
      println(y_col.t)
      println("y.ind: "+y_ind.t)
      */
      //Now compose the mini-batch (x,y) with whatever we could scrape
      val x = sparse(IMat(x_ind),IMat(x_col),FMat(x_val),this.dim,mb_size)
      val y = sparse(IMat(y_ind),IMat(y_col),FMat(y_val),this.dim,mb_size)
      return (x,y,docID)
    }
    System.err.println(" ERROR: Cannot draw from an empty sample-cache....")
    return null
  }

  def printPtrStats():String = {
    var out = ""
    var i = 0
    while(i < this.ptrs.size()){
      out += " === Doc " + i + "  ===\n"
      out += this.cache.get(ptrs.get(i)).printPtrStats() + "\n"
      out += " =====================\n"
      i += 1
    }
    return out
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
