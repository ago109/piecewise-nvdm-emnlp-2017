import java.util.{ArrayList, Random}
import java.util.HashMap

import YADLL.Utils.MiscUtils
import YAVL.Data.Text.Doc
import YAVL.Data.Text.Lexicon.Lexicon
import YAVL.DataStreams.Text.DocStream
import YAVL.Utils.Logger

/**
  * Code for converting a YADLL doc-stream to a set of document vectors, or rather, bag of words
  * (term frequency vectors). Note that this will apply a log(1 + TF) transform as this was what
  * Hinton did in his Replicated Softmax paper.
  *
  * Created by ago109 on 10/27/16.
  */
object StreamToSamples {

  def saveSamplesToDisk(fname : String, samples : ArrayList[DocSample]): Unit ={
    val fd = new Logger(fname)
    fd.openLogger()
    var i = 0
    while(i < samples.size()){
      val samp = samples.get(i).toString()
      fd.writeStringln(samp)
      i += 1
    }
    fd.closeLogger()
    println(" > Saved "+samples.size() + " to: "+fname)
  }

  def main(args : Array[String]): Unit = {
    if (args.length < 5) {
      System.err.println("usage: [/path/to/stream.txt] [seed] [tokenType] [/path/to/lexicon] [/path/to/subset.txt]" +
        " ?[/path/to/complement.txt] ?[numDocsInSubset] ")
      return
    }
    val fname = args(0)
    val seed = args(1).toInt
    val rng = new Random(seed)
    val tokenType = args(2)
    val fdict = args(3)
    val ftrain = args(4)
    var fvalid:String = null
    var numDocsInSubset:Int = 0
    if(args.length > 5){
      fvalid = args(5)
      numDocsInSubset = args(6).toInt
    }
    println(" > Loading symbol-lexicon...")
    val dict = new Lexicon(fdict)
    println(" > Reading stream...")
    //1st-pass gather pointers
    val samples = new ArrayList[DocSample]()
    val tmpStream = new DocStream(fname)
    tmpStream.setTokenType(tokenType)
    tmpStream.lowerCaseDocs(false)
    var idx = 0
    var doc: Doc = tmpStream.nextDoc()
    while (!tmpStream.atEndOfStream() || doc != null) {
      val idx_val_map = new HashMap[Integer,java.lang.Float]()
      while(doc.atEOF() == false){
        val symb = doc.getCurrentSymbol()
        val symb_idx = dict.getIndex(symb)
        if(symb_idx != null){
          var currVal = idx_val_map.get(symb_idx)
          if(currVal != null){
            currVal += 1f
            idx_val_map.put(symb_idx,currVal)
          }else{
            idx_val_map.put(symb_idx,1f)
          }
        }//else toss out symbols that do NOT occur in lexicon...
        doc.advanceSymbolPtr()
      }
      val sample = new DocSample(idx,dict.getLexiconSize())
      samples.add(sample)
      sample.buildFrom(idx_val_map, applyTransform = true)
      idx += 1
      print("\r " + samples.size() + " docs converted to bag-of-words...")
      doc = tmpStream.nextDoc() //grab next doc from Doc-Stream
    }
    println()

    //Now draw w/o replacement if so desired
    if(numDocsInSubset > 0){
      val subsetSamples = new ArrayList[DocSample]()
      var currNumInSubset = 0
      while(currNumInSubset < numDocsInSubset && samples.size() > 0){
        val ptr = MiscUtils.genRandInt(rng,0,samples.size())
        val samp = samples.remove(ptr) //draw document w/o replacement
        subsetSamples.add(samp)
        currNumInSubset += 1
      }
      //Now save subset & complement to disk
      StreamToSamples.saveSamplesToDisk(ftrain,samples)
      StreamToSamples.saveSamplesToDisk(fvalid,subsetSamples)
    }else{ //just dump all of the created samples to a single set on disk
      StreamToSamples.saveSamplesToDisk(ftrain,samples)
    }

  }


}
