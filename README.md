# clj-nd4j

This library is a wrapper around ND4J, the "Java Numpy".
The scope of this is mainly related to n-dimensional arrays, i.e.
scalar, vectors, matrices and tensors. It also contains datasets and
DL4J (ML) related material not to be used as standalone.
As of the intended DL4J design, this wrapper will try to offer a
pythonic flavour to make Java worth the shot.

[![Clojars Project](https://img.shields.io/clojars/v/io.github.jeremylcfr/clj-nd4j.svg)](https://clojars.org/io.github.jeremylcfr/clj-nd4j)
[![cljdoc badge](https://cljdoc.org/badge/io.github.jeremylcfr/clj-nd4j)](https://cljdoc.org/d/io.github.jeremylcfr/clj-nd4j)

## ND4J Overview

ND4J is a performance-oriented library based on native code using JavaCPP as
middleware. It also implements native BLAS libraries with the implementation you
want (Intel MKL or OpenBlas).

Obviously, this library is made for intensive programming and might be not worth the
investment for small applications. Apache Commons Math, for instance, is enough in most cases.
As a subset of DL4J, a java ML library,  this library would be also mandatory to build neural networks
and other ML applications using core DL4J functions.

See ND4J [official page](https://deeplearning4j.konduit.ai/nd4j/tutorials/quickstart).

## Design notes

This library includes clj-java-commons and so follows its naming conventions.
It uses also its coercion policy.
In short :
- primitive 1D or 2D arrays are favored and are not rebuild contrary to the
  general clojure philosophy
- arrowed functions ("->") are high level coercers (like ->nd-array) which conditionally
  check if input is already of target type

It also follows the "!" naming which indicates there that the function mutates its input
or does not return anything.

As a general rule, this library does not follow the Java codebase tree which is quite hard
to follow. We tried to centralize all usable material into clj-nd4j within core and dataset namespaces.
Other namespaces are either experimental or material for higher level modules (DL4J or Datavec).
Some performance is sacrificed for the sake of user-friendliness, provided that DL4J main source
of computation time is...actual machine learning procesing.

Features are added slowly in order to do not having a whole bunch of cryptic features which would
be terrible to maintain. We focus here on basic operations and will include new material as maturity
is built. Use directly java code if you need something we have skipped or missed !

It is also advised nor to use clj-nd4j.matrix namespace which is a bit silly but kept "in case".
This namespace will probably be trashed as soon as nd4j 1.0.0 will come.

## Project inclusion

The project is currently in ALPHA, content works but is not complete.
Additionaly, this library was developped with a quite old ND4J version and 
made compatible accross BREAKING changes. Thus, newer features were not implemented.

Finally, 0.X.Y versions will not be considered as real releases. Only 1.0.0 will start
the series with a pretty confident design. Main namespaces will probably not break, 
but it is not guaranteed. 

```clojure
[jeremylcfr/clj-nd4j "0.1.0-SNAPSHOT"]
```

You can additionaly add target implementation from ND4J like :

```clojure
[org.nd4j/nd4j-cuda-11.6 "1.0.0-M2.1"]
```
for CUDA (GPU), etc. 
See supported backends.

## Usage

Please find below a non-exhaustive documentation

### Starting a repl

A REPL namespaces is included under the `:dev` profile.
So just `lein repl` at project root.

Here is the current definition :

```clojure
(ns repl
  (:require [clj-nd4j [ndarray :as nda]
                      [dataset :as ndd]]
            [clj-java-commons [core :refer :all]
                              [coerce :refer [->clj]]])
  (:refer-clojure :exclude [/ neg? pos?]))
```

### Building nd-arrays

Nd-arrays can be built from different levels of automation.
However, even if it is visually less intuitive, its is encouraged to
use the native ND4J way(s). Indeed, the most generic constructor uses
a flattened representation paired with shape description describing
the dimensions of the cross-product involved unless data coercion is
neglectible compared to your whole application runtime.

```clojure
;; Generic one
(nd-array [2 3] [1 2 3 4 5 6]) ;; ~ matrix [[1 2] [3 4] [5 6]]

;; Automated version, this uses custom Clojure code to infer shape...
;; (the inference is unsafe to do not sacrifice too much performance, i.e.
;;  only first elements of each nested array are used)
(nd-array [[1 2] [3 4] [5 6]])

;; ...however, hopefully, classical shapes are natively supported
;; You just need to "hint" this within this wrapper, this will call
;; the right Java method
(nd-array :array [1 2 3 4]) ;; ~ array [1 2 3 4]
(nd-array :matrix [[1 2] [3 4] [5 6]]) ;; ~ matrix [[1 2] [3 4] [5 6]]
;; Or you can use the subset clj-nd4j.matrix, however this namespace might disappear

;; Use "->" version for generic functions targetting users
;; It accepts nd-arrays as input and does nothing to them
(->nd-array [2 3] [1 2 3 4 5 6])

;; Even more general for data-oriented applications
(->nilable-nd-array nil) ;; does not throw and returns nil

;; Specific constructors are also provided
(nd-zeros [2 3])
(nd-ones [2 3])
(nd-rand [2 3])
```

### Self operations

Basic self- and inter- operations are implemented :

```clojure
;; Let(s define a matrix
(def m (nd-array [2 3] [1 2 3 4 5 6]))

;; Getters are included
;; BEWARE : you need good visual capabilities :)
(get-in-scalar m [0 1]) ;; [2] as a NDArray
(get-in-double m [0 1] ;; 2.0
(get-double m 3) ;; 4.0 (this is the index of the flattened array)
(get-row m)

;; Basic self-operations
(transpose m)
(roll-axis m 1) ;; not that basic, see explanation from NumPy https://docs.scipy.org/doc/numpy/reference/generated/numpy.rollaxis.html
```

Regarding mutability, as a general rule all functions do not alter the NDArray.
However, some functions can alters the current instance, they are named with a
BANG (!) following the clojure specification

```clojure
;; First, you can clone you nd-array
(def m* (clone m))

;; Let's normalize (in the statistical meaning)
(normalize m)
(= m m*) ;; true
;; Now...
(normalize! m)
( = m m*) ;; false, m has been altered

;; Thus, a lot of element-wise functions are provided
;; with mutable and immutable versions
(abs! m)
(relu6 m)
(exp! m)
(softmax m)
...

;; Here is the whole list...
;; The BANG version is exactly the same with !
(def single-ops
  {:negate negate
   :abs abs
   :floor floor
   :ceil ceil
   :sqrt sqrt
   :round round
   :stabilize stabilize
   :exp exp
   :cosh cosh
   :sinh sinh
   :tanh tanh
   :hard-tanh hard-tanh
   :dhard-tanh dhard-tanh
   :atanh atanh
   :ln ln
   :log log
   :sign sign
   :cos cos
   :sin sin
   :acos acos
   :asin asin
   :atan atan
   :sigmoid sigmoid
   :dsigmoid dsigmoid
   :eps eps
   :softmax softmax
   :softplus softplus
   :softsign softsign
   :dsoftsign dsoftsign
   :relu relu
   :relu6 relu6
   :leaky-relu leaky-relu
   :dleaky-relu dleaky-relu
   :elu elu
   :delu delu})

;;...from which you can call the function
;; This is useful when you want to search the right function within
;; the pool with a meta-algorithm
(transform :exp m)
(transform! :exp m)

;; And give back clojure
;; Works for any nd-array
(->clj m) ;; [[1 2] [3 4] [5 6]]
```

### Co-operators

Co-operators are also defined, from basic to advanced ones :

```clojure
;; Basic operations
;; Some like mmul are included within matrix namespace
;; This will probably change
(t* m m*) ;; tensor multiplication

;; Distances
(manhattan-distance m m*) ;; this is L1
(jaccard-distance m m*)
;; or
(distance :manhattan m m*)

;; Mutable and immutable
(pow m m*)
(pow! m m*)
(cotransform :pow m m*)
```

### I/O gimmick

```clojure
(write-nd-array! "my-arrray.csv" m)
(def m** (load-nd-array m))
```

### DL4J material

The following material is mostly for DL4J use and has no interest within the scope
of this library alone. Especially, ML-related stuff will not be covered.
Dataset usage is fairly simple :

```clojure
;; Basic usage
(def labels (nd-array :array [1.0 2.0 3.0]))
(def features (nd-array :array [1.0 4.0 9.0]))
(dataset features labels)

;; Organized version (slower)
(->dataset {:labels [1 2 3] :features [1 4 9]})
~
(->dataset [[1 1] [2 4] [3 9]]) ;; [x y] approach
```

You can then interact in a basic way with it

```clojure
(do-scale! d) ;; mutable AND returns void
(scale! d) ;; muatble AND returns d
(multiply! m 2) ;; guess...
(binarize d) ;; immutable operation
```

Finally, you can write datasets and read them back.

```clojure
;; Write
(write-dataset! "dataset.txt" d)

;; Read
;; Now you must be familiar with notations
;; though 1-arity versions are equal
(read-dataset! "dataset.txt")
(read-dataset! "dataset.txt" d)
(read-dataset "dataset.txt")
(read-dataset "dataset.txt" d)
```

Read code and DL4J (or incubated clj-dl4j) for more info.

## Dependencies

As a basic low-level library, clj-nd4j has only ND4J related material and clj-java-commons.
No dependencies should be included apart from clojure.core/Java lang/util/etc. ones.
Small utilities must be copy-pasted/rewritten.

## License

Copyright © 2022 Jérémy Le Corguillé

Apache License 2.0
