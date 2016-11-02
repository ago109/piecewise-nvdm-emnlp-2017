import java.util.HashMap

/**
  * Simple container to represent a bag-of-words model for a labeled document. NOTE that the toString()
  * method of this object will print the sparse bag of words vector to libsvm format:<br>
  *   [doc_idx] [feat_idx_0]:[feat_val_0] ... [feat_idx_n]:[feat_val_n]
  * <br>
  * Created by ago109 on 10/27/16.
  */
class DocSample(var doc_id: Int, var dim : Int, var bagOfIdx : Array[Int] = null, var bagOfVals : Array[Float] = null) {
  var targetPtr = 0
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

  def drawTargetIdx():Int ={
    var targ = -1
    if(this.targetPtr < this.bagOfIdx.length) {
      targ = this.bagOfIdx(targetPtr)
      this.targetPtr += 1
    }
    return targ
  }

  def isDepleted(): Boolean ={
    if(this.targetPtr >= this.bagOfIdx.length) {
      return true
    }
    return false
  }

  def resetTargetIdx(): Unit ={
    this.targetPtr = 0
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
      ptr += 1
    }
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
