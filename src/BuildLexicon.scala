import YAVL.TextStream.Dict.Lexicon
import YAVL.TextStream.{Doc, TextStream}

/**
  * Builds a symbol lexicon from a doc-stream (or "corpus").
  *
  * Created by ago109 on 9/4/16.
  * External args: -Djava.library.path=/home/ago109/workspace/BIDMat/lib/
  */
object BuildLexicon {
  def main(args: Array[String]): Unit = {
    if(args.length < 4){
      System.err.println("usage: [/path/to/stream.txt] [/path/to/lexicon_out] [delim] [minTermFreq]" +
        " ?[startToken] ?[endToken] ?[oovToken]")
      return
    }
    val dataFname = args(0)
    val dictFname = args(1)
    val delim = args(2)
    val minTermFreq = args(3).toInt
    var startTok:String = null
    var endTok:String = null
    var oovToken:String = null
    if(args.length > 4){
      startTok = args(4)
    }
    if(args.length > 5){
      endTok = args(5)
    }
    if(args.length > 6){
      oovToken = args(6)
    }

    val stream = new TextStream(dataFname) //prop up stream reader
    val dict = new Lexicon()
    var numDocsProcessed = 0
    var doc:Doc = null
    while (!stream.atEOS()) {
      doc = stream.nextDoc()
      if (doc != null) {
        numDocsProcessed = numDocsProcessed + 1
        dict.updateDict(doc.text.split(delim))
      }
      print("\r > # Docs Processed = " + numDocsProcessed)
    }
    println()
    //Do any pruning & save final dictionary to disk...
    if (minTermFreq > 0) {
      dict.pruneDict(minTermFreq) //<- prune out symbols that occur < minTermFreq : Int
      //dict.addUnknownToken("OOV")
    }
    println(" > Saving built lexicon: " + dictFname)
    //dict.saveDictionary(dictFname)
    dict.startTok = startTok
    dict.endTok = endTok
    dict.oovTok = oovToken
    dict.removeSymbol(dict.startTok) // <-- we do NOT want start-token included in input dimension
    dict.saveLexicon(dictFname)
    dict.saveDictionary(dictFname + "_text.dict")
    println(" Lexicon.size = "+dict.getLexiconSize())
  }
}