import YAVL.Data.Text.Doc
import YAVL.Data.Text.Lexicon.Lexicon
import YAVL.DataStreams.Text.DocStream
import YAVL.Utils.ConfigFile

/**
  * Builds a symbol lexicon from a doc-stream (or "corpus").
  *
  * Created by ago109 on 9/4/16.
  * External args: -Djava.library.path=/home/ago109/workspace/BIDMat/lib/
  */
object BuildLexicon {
  def main(args: Array[String]): Unit = {
    if(args.length != 4){
      System.err.println("usage: [/path/to/stream.txt] [/path/to/lexicon_out.dict] [tokenType]" +
        " [minTermFreq]")
      return
    }
    val dataFname = args(0)
    val dictFname = args(1)
    val tokenType = args(2)
    val minTermFreq = args(3).toInt

    val stream = new DocStream(dataFname) //prop up stream reader
    stream.setTokenType(tokenType);
    stream.lowerCaseDocs(true) //lower-case all streaming text/symbols
    val dict = new Lexicon()
    var numDocsProcessed = 0
    var doc:Doc = null
    while (stream.atEndOfStream() == false) {
      doc = stream.nextDoc()
      if (doc != null) {
        numDocsProcessed = numDocsProcessed + 1
        dict.updateDict(doc)
      }
      print("\r > # Docs Processed = " + numDocsProcessed)
    }
    if (doc != null) {
      numDocsProcessed = numDocsProcessed + 1
      dict.updateDict(doc)
      print("\r > # Docs Processed = " + numDocsProcessed)
    }
    println()
    //Do any pruning & save final dictionary to disk...
    if (minTermFreq > 0) {
      dict.pruneDict(minTermFreq) //<- prune out symbols that occur < minTermFreq : Int
      //dict.addUnknownToken("OOV")
    }
    println(" > Saving built lexicon: " + dictFname)
    dict.saveDictionary(dictFname)
  }
}
