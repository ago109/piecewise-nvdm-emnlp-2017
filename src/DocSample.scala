import java.util.{ArrayList, HashMap, Random}

import YADLL.Utils.MiscUtils

/**
  * Simple container to represent a bag-of-words model for a labeled document. NOTE that the toString()
  * method of this object will print the sparse bag of words vector to libsvm format:<br>
  *   [doc_idx] [feat_idx_0]:[feat_val_0] ... [feat_idx_n]:[feat_val_n]
  * <br>
  * Created by ago109 on 10/27/16.
  */
class DocSample(var doc_id: Int, var dim : Int, var bagOfIdx : Array[Int] = null, var bagOfVals : Array[Float] = null) {
  var ptrs = new ArrayList[Int]()
  var cnts = new ArrayList[Int]() //maps 1-to-1 w/ ptrs
  var depletedPtrs = new ArrayList[Int]()
  var depletedCnts = new ArrayList[Int]()
  var docLen = 0f

  def getMinTermValue():Float={
    var min = 10000f
    var t = 0
    while(t < bagOfVals.length){
      min = Math.min(min,bagOfVals(t))
      t += 1
    }
    return min
  }

  def getMaxTermValue():Float={
    var max = 0f
    var t = 0
    while(t < bagOfVals.length){
      max = Math.max(max,bagOfVals(t))
      t += 1
    }
    return max
  }

  def drawTargetIdx(rng : Random):Int ={
    var idx = 0
    if(rng != null){
      idx = MiscUtils.genRandInt(rng,0,this.ptrs.size())
    }
    var cnt = this.cnts.get(idx)
    cnt -= 1
    this.cnts.set(idx,cnt)
    if(cnt <= 0){ //if count for this word has been set to 0, this word has been exhaustively sampled
      this.cnts.remove(idx)
      idx = this.ptrs.remove(idx)
      this.depletedPtrs.add(idx)
    }else{
      idx = this.ptrs.get(idx)
    }
    return idx
  }

  def isDepleted(): Boolean ={
    if(this.ptrs.size() == 0) {
      return true
    }
    return false
  }

  def resetTargetIdx(): Unit ={
    this.ptrs = this.depletedPtrs
    this.cnts.addAll( this.depletedCnts )
    this.depletedPtrs = new ArrayList[Int]()
  }

  /**
    * Build a document-sample from an index-value map.
    *
    * @param idx_val_map
    * @param applyTransform -> apply log(1 + TF) transform to values? (rounded to nearest integer)
    */
  def buildFrom(idx_val_map : HashMap[Integer,java.lang.Float], applyTransform : Boolean = false,
                binarizeVectors : Boolean = false): Unit ={
    this.bagOfIdx = new Array[Int](idx_val_map.size())
    this.bagOfVals = new Array[Float](idx_val_map.size())
    val iter = (idx_val_map.keySet()).iterator()
    var ptr = 0
    while (iter.hasNext) {
      val idx:Int = iter.next()
      var value:Float = idx_val_map.get(idx)
      docLen += value
      if(binarizeVectors){ //Simple binarization filter
        value = 1f
      }else if(applyTransform){ //Hinton-style log(1 + TF) filter
        value = Math.log(1f + value).toFloat
        if(value >= 0.5f){
          value = Math.round(Math.log(1f + value).toFloat)
        }else{ //lower bound to 1 so no extra zero values are created
          value = 1f
        }
      }
      this.bagOfIdx(ptr) = idx
      this.bagOfVals(ptr) = value
      this.ptrs.add(idx)
      this.cnts.add(value.toInt)
      ptr += 1
    }
    this.depletedCnts.addAll( this.cnts )
  }

  /**
    * A special routine to ensure print-out of 1-based indices (like proper libsvm format).
    * @return
    */
  def toLibSVMString():String ={
    var out = "" + this.doc_id + " "
    var i = 0
    while(i < this.bagOfIdx.length){
      out += ""+ (this.bagOfIdx(i)+1) + ":" + this.bagOfVals(i)+" "
      i += 1
    }
    out = out.substring(0,out.length()-1) //nix trailing space...
    return out
  }

  override def toString():String ={
    var out = "" + this.doc_id + " "
    var i = 0
    while(i < this.bagOfIdx.length){
      out += ""+this.bagOfIdx(i) + ":" + this.bagOfVals(i)+" "
      i += 1
    }
    out = out.substring(0,out.length()-1) //nix trailing space...
    return out
  }

}
