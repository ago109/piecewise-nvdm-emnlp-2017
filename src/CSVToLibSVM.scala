import YADLL.Utils.MiscUtils
import YAVL.Utils.Logger

/**
  * Transforms a dense CSV file with many zeros to libsvm sparse format (1-based indexing)
  *
  * Created by ago109 on 11/4/16.
  */
object CSVToLibSVM {

  def main(args : Array[String]): Unit ={
    if(args.length != 3){
      System.err.println("usage: [/path/to/file.csv] [/path/to/out.vec] [appendIDs?]")
      return
    }
    val fname = args(0)
    val out_fname = args(1)
    val appendIDs = args(2).toBoolean
    val wd = new Logger(out_fname)
    wd.openLogger()
    println(" > Loading " + fname)
    var numDiscards = 0
    val fd = MiscUtils.loadInFile(fname)
    var line = fd.readLine()
    var idx = 0
    while(line != null){
      var outLine = ""
      val tok = line.split(",")
      var numNonZeros = 0
      if(appendIDs == false){
        outLine += tok(0) + " "
        var i = 1
        while(i < tok.length){
          if(tok(i).toFloat != 0f) {
            outLine += (i+1) + ":" + tok(i) + " "
            numNonZeros += 1
          }
          i += 1
        }
      }else{
        outLine += idx + " "
        var i = 0
        while(i < tok.length){
          if(tok(i).toFloat != 0f){
            outLine += (i+1) + ":" + tok(i) + " "
            numNonZeros += 1
          }//else, omit zero values
          i += 1
        }
      }
      if(numNonZeros > 0) {
        wd.writeStringln(outLine.substring(0, outLine.length() - 1)) //trim off excess space @ end
        idx += 1
        print("\r > Converted " + idx + " lines to libsvm format...")
      }else{
        numDiscards += 1
      }
      line = fd.readLine()
    }
    wd.closeLogger()
    fd.close()
    println()
    println(" > Wrote output to "+out_fname)
    println(" > Discarded "+ numDiscards + " empty docs...")
  }


}
