import java.util.ArrayList

import YADLL.Utils.MiscUtils

import scala.collection.immutable.List
import scala.collection.JavaConversions._
import scala.collection.JavaConverters._

/**
  * Very specialized code meant to transform the RCV1 term map list and select only the k lowest
  * IDF-ranked words, also effectively rebuilding the document vectors for the streams targeted.
  *
  * Created by ago109 on 11/5/16.
  */
object RCV1_LexBuilder {

  //A Reuters record has a word, its integer in feature-space, and an inverse doc-freq score
  case class Record(var symbol:String, var idx:Int, var idf:Float)

  def main(args : Array[String]): Unit ={
    val fname = args(0)
    val outFname = args(1)
    val k = args(2).toInt

    //Reuter term file reads: term <id> <idf_wght>
    val recs = new ArrayList[Record]()

    val fd = MiscUtils.loadInFile(fname)
    var line = fd.readLine()
    while(line != null){
      //Parse line
      val tok = line.split("\\s+")
      val term = tok(0)
      val idx = tok(1).toInt
      val idf = tok(2).toFloat
      val rec = new Record(term,idx,idf)
      recs.add(rec)
      line = fd.readLine()
    }
    val comp = (z:Record) => {z.idf} //comparator (really, just a literal that pulls out that inverse doc-freq)
    val ll = recs.asScala.toList
    ll.sortBy(comp) //sort list by idf scores
    //ll.foreach(
     // (Nothing) => 1
    //)


  }

}
