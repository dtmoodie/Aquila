We want a tensor library primarily for container purposes to densly store arrays, matricies, etc.
Want to have a data type that can wrap an image and that can natively wrap strided / padded images.

Evaluation Criteria:
- Support
 - updates / bufxies
 - GPU compute
 - mathematical operations
 - template expressions
- API usability
 - wrap data
 - clear semantics and usage
 -
- whiteboxing (can we wrap other data in this api)
 - dense, and strided data

For this purpose we have so far evaluated:
Eigen::Tensors
pros:
- Well supported with several mathematical functions and the full backing of Eigen
- Has device functionality
- OG template expressions
- eigen matrix interop
cons:
- Does not support wrapping strided images, the best I can think of with this is to create a tensor map of the full image
  and then return a strided view or a slice, but that returns an operation not another tensor.  We would then need to
  pass around tensor references everywhere.  This may be possible so this may need more exploration. Maybe we should just copy strided data?
- col major


Tensorflow's tensor (https://www.tensorflow.org/api_docs/cc/class/tensorflow/tensor)
pros:
- Higher level API ontop of an Eigen::Tensor
cons:
- more dependencies, requires protobuf
- has a ton of extra stuff

dlib (https://github.com/davisking/dlib)
cons:
- only 4d, designed strictly for batched images
- looks to have only one possible layout


XTensor (https://github.com/xtensor-stack/xtensor)
pros:
- Lazy eval
- row major by default
- can wrap data
cons:
- no GPU support



mxnet's mshadow (https://github.com/apache/incubator-mxnet/tree/master/3rdparty/mshadow)
cons
- not a standalone repo
pros:
- pretty good API with template expressions

pytorch's ATen (https://github.com/pytorch/pytorch/tree/master/aten/src)
cons
- not standalone


ASTen (https://github.com/ZichaoLong/ASTen)


Fastor (https://github.com/romeric/Fastor)
pros:
- supports template expressions
- compile time operation optimization
- can wrap data
cons:
- only compile time shapes :(


TACO (https://github.com/tensor-compiler/taco)

Armadillo (http://arma.sourceforge.net/)


NDArray (https://github.com/ndarray/ndarray)


Tensor Comprehensions (https://github.com/facebookresearch/TensorComprehensions)

minitensor
-
