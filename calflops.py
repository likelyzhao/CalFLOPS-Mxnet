


def calConvFlops(inshape,outshape,kernel_size,hasbias = True, num_group = 1, type='normal'):
    # inshape is [chanel , width , height]
    # kernel_size in [kernelwidth , kernelheight]
    # outshape is [chanel * width * height]
    # type = 'normal' 'depthwise' 'deformable'
    # return [multipflops,addflops,compare flops,expflops]
    addflops =0
    multipflops = inshape[0] * outshape[0] * outshape[1] * \
            outshape[2] * kernel_size[0] * kernel_size[1] / num_group
    if hasbias:
        addflops =  outshape[0] * outshape[1] * outshape[2]

    return [multipflops,addflops,0,0]

def calActivationFlops(inshape,outshape,type='relu'):
    # inshape is [chanel , width , height]
    # outshape is [chanel * width * height]
    # type = 'relu' 'sigmoid' 
    # return [multipflops,addflops,compare flops,expflops]
    compareflops =1
    if type == 'relu':
        for shape in outshape:
          compareflops *= shape

    return [0,0,compareflops,0]

def calPoolingFlops(inshape,outshape,kernel_size,type='max'):
    # inshape is [chanel , width , height]
    # kernel_size in [kernelwidth , kernelheight]
    # outshape is [chanel * width * height]
    # type = 'max' 'ave' 'gop' 'deformable'
    # return [multipflops,addflops,compare flops,expflops]

    compareflops =0
    addflops =0
    multipflops =0

    if type == 'max':
        compareflops += outshape[0] * outshape[1] * outshape[2] * kernel_size[0] * kernel_size[1]
    
    if type == 'ave':
        multipflops += outshape[0] * outshape[1] * outshape[2] 
        addflops += outshape[0] * outshape[1] * outshape[2] * kernel_size[0] * kernel_size[1]

    if type == 'gop':
        multipflops += outshape[0] * outshape[1] * outshape[2]
        addflops += outshape[0] * outshape[1] * outshape[2] * inshape[0] * inshape[1]


    return [multipflops,addflops,compareflops,0]

def calFcFlops(inshape,outshape,hasbias = True):
    # inshape is [Inchanel]
    # outshape is [Outchanel]
    # type = 'max' 'ave' 'gop'
    # return [multipflops,addflops,compare flops,expflops]

    addflops =0
    multipflops = inshape[0] * outshape[0]

    if hasbias :
        addflops +=  outshape[0]
    
    return [multipflops,addflops,0,0]

'''

def numOfBytes(dtype: DType.DType): Int = {
    dtype match {
      case DType.UInt8 => 1
      case DType.Int32 => 4
      case DType.Float16 => 2
      case DType.Float32 => 4
      case DType.Float64 => 8
      case DType.Unknown => 0
    }
  }
  
  def str2Tuple(str: String): List[String] = {
    val re = """\d+""".r
    re.findAllIn(str).toList
  }
  
  def main(args: Array[String]): Unit = {
    val cafl = new CalFlops
    val parser: CmdLineParser = new CmdLineParser(cafl)
    try {
      parser.parseArgument(args.toList.asJava)
      
      assert(cafl.symbol != "")
      assert(cafl.dataShapes.length > 0)
      
      println(s"${cafl.symbol} ${cafl.dataShapes.mkString(", ")} ${cafl.labelShapes.length}")
      
      val network = Symbol.load(cafl.symbol)
      
      val dataShapes = cafl.dataShapes.map { s =>
        val tmp = s.trim().split(",")
        val name = tmp(0)
        val shapes = Shape(tmp.drop(1).map(_.toInt))
        DataDesc(name = name, shape = shapes)
      }.toIndexedSeq
      
      val dataNames = dataShapes.map(_.name)
      
      val (labelShapes, labelNames): (scala.Option[IndexedSeq[DataDesc]], IndexedSeq[String]) = {
        if (cafl.labelShapes.length == 0) {
          (None, null)
        } else {
          val shapes = cafl.labelShapes.map { s =>
            val tmp = s.trim().split(",")
            val name = tmp(0)
            val shapes = Shape(tmp.drop(1).map(_.toInt))
            DataDesc(name = name, shape = shapes)
          }.toIndexedSeq
          val names = shapes.map(_.name)
          (Some(shapes), names)
        }
      }
      
      val argss = network.listArguments()
      assert(labelNames.forall(n => argss.contains(n)))
      
      val module = new Module(network, dataNames, labelNames)
      module.bind(dataShapes, labelShapes, forTraining = false)
      module.initParams()

      val (argParams, auxParams) = module.getParams
      
      
      var totalFlops = 0f
      
      val conf = JSON.parseFull(network.toJson) match {
        case None => null
        case Some(map) => map.asInstanceOf[Map[String, Any]]
      }
      require(conf != null, "Invalid json")
      require(conf.contains("nodes"), "Invalid json")
      val nodes = conf("nodes").asInstanceOf[List[Any]]
      
      nodes.foreach { node =>
        val params = node.asInstanceOf[Map[String, Any]]
        val op = params("op").asInstanceOf[String]
        val name = params("name").asInstanceOf[String]
        val attrs = {
          if (params.contains("attrs")) params("attrs").asInstanceOf[Map[String, String]]
          else if (params.contains("param")) params("param").asInstanceOf[Map[String, String]]
          else Map[String, String]()
        }
        
        val inputs = params("inputs").asInstanceOf[List[List[Double]]]
        
        op match {
          case "Convolution" => {
            val internalSym = network.getInternals().get(name + "_output")
            val (internalLabelShapes, internalLabelNames) = getInternalLabelInfo(internalSym, labelShapes)
            
            val tmpModel = new Module(internalSym, dataNames, internalLabelNames)
            tmpModel.bind(dataShapes, internalLabelShapes, forTraining = false)
            tmpModel.initParams()
            val outShape = tmpModel.getOutputsMerged()(0).shape
            
            // support conv1d NCW and conv2d NCHW layout
            val outShapeProdut = if (outShape.length == 3) outShape(2) else outShape(2) * outShape(3)
            totalFlops += outShapeProdut * argParams(name + "_weight").shape.product * outShape(0)
            
            if (argParams.contains(name + "_bias")) {
              totalFlops += outShape.product
            }
          }
          case "Deconvolution" => {
            val inputLayerName = {
              val inputNode = nodes(inputs(0)(0).toInt).asInstanceOf[Map[String, Any]]
              inputNode("name").asInstanceOf[String]
            }
            
            val internalSym = network.getInternals().get(inputLayerName)
            val (internalLabelShapes, internalLabelNames) = getInternalLabelInfo(internalSym, labelShapes)

            val tmpModel = new Module(internalSym, dataNames, internalLabelNames)
            tmpModel.bind(dataShapes, internalLabelShapes, forTraining = false)
            tmpModel.initParams()
            val inputShape = tmpModel.getOutputsMerged()(0).shape

            totalFlops += inputShape(2) * inputShape(3) * argParams(name + "_weight").shape.product * inputShape(0)
            
            if (argParams.contains(name + "_bias")) {
              val internalSym = network.getInternals().get(name + "_output")
              val (internalLabelShapes, internalLabelNames) = getInternalLabelInfo(internalSym, labelShapes)
              
              val tmpModel = new Module(internalSym, dataNames, internalLabelNames)
              tmpModel.bind(dataShapes, internalLabelShapes, forTraining = false)
              tmpModel.initParams()
              val outShape = tmpModel.getOutputsMerged()(0).shape
              
              totalFlops += outShape.product
            }
          }
          case "FullyConnected" => {
            totalFlops += argParams(name + "_weight").shape.product * dataShapes(0).shape(0)
            if (argParams.contains(name + "_bias")) {
              val numFilter = argParams(name + "_bias").shape(0)
              totalFlops += numFilter * dataShapes(0).shape(0)
            }
          }

          case "Pooling" => {
            val globalPool = {
              if (!attrs.contains("global_pool")) false
              else if (attrs("global_pool") == "False") false
              else true
            }
            if (globalPool) {
              val inputLayerName = {
                val inputNode = nodes(inputs(0)(0).toInt).asInstanceOf[Map[String, Any]]
                inputNode("name").asInstanceOf[String]
              }

              val internalSym = network.getInternals().get(inputLayerName + "_output")
              val (internalLabelShapes, internalLabelNames) = getInternalLabelInfo(internalSym, labelShapes)

              val tmpModel = new Module(internalSym, dataNames, internalLabelNames)
              tmpModel.bind(dataShapes, internalLabelShapes, forTraining = false)
              tmpModel.initParams()
              val inputShape = tmpModel.getOutputsMerged()(0).shape
  
              totalFlops += inputShape.product
            } else {
              val internalSym = network.getInternals().get(name + "_output")
              val (internalLabelShapes, internalLabelNames) = getInternalLabelInfo(internalSym, labelShapes)
              
              val tmpModel = new Module(internalSym, dataNames, internalLabelNames)
              tmpModel.bind(dataShapes, internalLabelShapes, forTraining = false)
              tmpModel.initParams()
              val outShape = tmpModel.getOutputsMerged()(0).shape
              val kernelP = str2Tuple(attrs("kernel")).map(_.toInt).reduce(_ * _)
              
              totalFlops += outShape.product * kernelP
            }
          }
          case "BatchNorm" => {}
          case "Activation" => {
            attrs("act_type") match {
              case "relu" => {
                val inputLayerName = {
                  val inputNode = nodes(inputs(0)(0).toInt).asInstanceOf[Map[String, Any]]
                  inputNode("name").asInstanceOf[String]
                }
  
                val internalSym = network.getInternals().get(inputLayerName + "_output")
                val (internalLabelShapes, internalLabelNames) = getInternalLabelInfo(internalSym, labelShapes)
  
                val tmpModel = new Module(internalSym, dataNames, internalLabelNames)
                tmpModel.bind(dataShapes, internalLabelShapes, forTraining = false)
                tmpModel.initParams()
                val inputShape = tmpModel.getOutputsMerged()(0).shape
                
                totalFlops += inputShape.product
              }
              case _ => {}
            }
          }
          case "LeakyReLU" => {}
          case _ => {}
        }
      }
      
      val totalSize = {
        val argSize = (0f /: argParams){ (size, elem) =>
          size + elem._2.shape.product * numOfBytes(elem._2.dtype)
        }
        val auxSize = (0f /: auxParams){ (size, elem) =>
          size + elem._2.shape.product * numOfBytes(elem._2.dtype)
        }
        argSize + auxSize
      }


    println(s"flops: ${totalFlops / 1000000} MFLOPS")
    println(s"model size: ${totalSize / 1024 / 1024} MB")
      
    } catch {
      case ex: Exception => {
        println(ex.getMessage, ex)
        parser.printUsage(System.err)
        System.exit(1)
      }
    }
  }
}

class CalFlops {
  @Option(name = "--ds", handler = classOf[StringArrayOptionHandler],
      required = true, usage = "the network json file to calculate flops.")
  private val dataShapes: Array[String] = Array[String]()
  @Option(name = "--ls", handler = classOf[StringArrayOptionHandler],
      usage = "the network json file to calculate flops.")
  private val labelShapes: Array[String] = Array[String]()
  @Option(name = "--symbol", usage = "the network json file to calculate flops.")
  private val symbol: String = ""
}
'''