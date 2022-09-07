(ns clj-nd4j.ndarray
  (:require [clj-java-commons.core :refer :all]
            [clj-java-commons.coerce :refer [->clj]])
  (:import [org.nd4j.linalg.factory Nd4j]
           [org.nd4j.linalg.api.buffer DataType]
           [org.nd4j.linalg.api.ndarray INDArray]
           [org.nd4j.linalg.ops.transforms Transforms]
           [org.nd4j.linalg.api.ops.impl.transforms Pad Pad$Mode])
  (:refer-clojure :exclude [/ pos? neg? abs]))

;; https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/factory/Nd4j.java#L3642
;; https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-buffer/src/main/java/org/nd4j/linalg/api/buffer/DataType.java


;; SELECTED TYPE

(def ->data-type
  {:double  DataType/DOUBLE
   :float   DataType/FLOAT
   :boolean DataType/BOOL
   :string  DataType/UTF8
   :int     DataType/INT32
   :long    DataType/INT64})

(def default-data-type DataType/DOUBLE)

(def ->pad-mode
  {:constant  Pad$Mode/CONSTANT
   :reflect   Pad$Mode/REFLECT
   :symmetric Pad$Mode/SYMMETRIC})

(def default-pad-mode Pad$Mode/CONSTANT)

;;////////////////////////////////////////////////////////////////////////////
;;============================================================================
;;                                 PREDICATES
;;============================================================================
;;\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

(defn nd-array?
  [obj]
  (instance? INDArray obj))

;;////////////////////////////////////////////////////////////////////////////
;;============================================================================
;;                                  BUILDERS
;;============================================================================
;;\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

;;=================================================
;;=======================RAW=======================
;;=================================================

;;=================Internal===============

(defn- unsafe-infer-shape
  "Internal usage.
   Infers shape from clojure model.
   Unsafe in a way that it assumes that your
   structure is normalized"
  [data]
  (loop [current-element data
         shape (transient [])]
    (if (sequential? current-element)
      (recur (first current-element) (conj! shape (count current-element)))
      (persistent! shape))))

;;=================Executor===============

(defn nd-array
  "Creates a nd array (n dimensional array) from data.
   For predictable input, indicating a shape (dimensions) is wiser.
   Else, shape will be infered from a walk from every first element
   unless input is
   When you indicate a shape, data must be provided flattened, else
   the structure must be already formatted as a nested structure.
   The sole exceptions to this rules are when you want to create
   a matrix or a (1D) array (most common cases). There you can replace
   shape by a type hint : :array or :matrix.
   You can also use specialized constructors : matrix and array
   Example :
   (nd-array [2 2] [1 2 3 4]) ;;  or (->nd-array [[1 2] [3 4]]) or (->nd-array :matrix [[1 2] [3 4]])
   ~
   [[1 2]
    [3 4]] (a matrix)
   (nd-array [4] [1 2 3 4]) ;;  (or (->nd-array [1 2 3 4]))
   ~
   [1 2 3 4] (an array)"
  ^INDArray
  ([data]
   (nd-array (unsafe-infer-shape data) (flatten data)))
  ([shape data]
   (if (sequential? shape)
     (Nd4j/create ^doubles (->double-array data) ^longs (->long-array shape))
     (case shape
           :array (Nd4j/create ^doubles (->double-array data))
           :matrix (Nd4j/create #^"[[D" (->ddouble-array data))
           (throw (Exception. (str "ND-ARRAY - Input shape " shape " is not valid, see documentation")))))))

;;=====================================================
;;=======================GENERIC=======================
;;=====================================================

(defn ->nd-array
  "Like nd-array constructor but also works for
   already existing NDArray instances (do nothing).
   Useful mostly internally to allow flexibility but
   can be used to design external generic functions
   as well"
  ^INDArray
  [obj]
  (if (nd-array? obj)
    obj
    (nd-array obj)))

;;=====================================================
;;=======================NILABLE=======================
;;=====================================================

(defn ->nilable-nd-array
  "Internal usage.
   Also deal with nil"
  ^INDArray
  [obj]
  (if (nd-array? obj)
    obj
    (if (nil? obj)
      nil
      (nd-array obj))))

;;=====================================================
;;=======================SPECIAL=======================
;;=====================================================

;;=================Zeros===============

(defn nd-zeros
  "Creates a NDArray full of zeros of
   input shape.
   Example :
   (nd-zeros [2 2]) ~ [[0 0] [0 0]]"
  ([shape]
   (Nd4j/zeros (->long-array shape)))
  ([data-type shape]
   (Nd4j/zeros (->data-type data-type) (->long-array shape))))

;;=================Ones===============

(defn nd-ones
  "Creates a NDArray full of ones of
   input shape.
   Example :
   (nd-ones [2 2]) ~ [[1 1] [1 1]]"
  ([shape]
   (Nd4j/ones (->long-array shape)))
  ([data-type shape]
   (Nd4j/ones (->data-type data-type) (->long-array shape))))

;;=================Rand===============

(defn nd-rand
  "Creates a NDArray initialized with random
   values (between 0 and 1) and of input shape."
  [shape]
  (Nd4j/rand (->long-array shape)))

;;=======================================================
;;=======================FROM FILE=======================
;;=======================================================

(defn load-nd-array
  "Load a NDArray from a file written with
   write-nd-array!"
  [path]
  (Nd4j/readTxt path))

;;////////////////////////////////////////////////////////////////////////////
;;============================================================================
;;                                  GETTERS
;;============================================================================
;;\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

;;=============================================================
;;=======================CHARACTERISTICS=======================
;;=============================================================

(defn get-shape
  "Gets shape of a NDArray as long[]"
  ^longs
  [^INDArray nd-array]
  (.shape nd-array))

;;==================================================
;;=======================DATA=======================
;;==================================================

;;=================Scalar===============

(defn get-scalar
  "Gets the nth leaf element of the flattended nd-array
   (line first) as a scalar nd-array (0d-array)"
  ^INDArray
  [^INDArray nd-array n]
  (.getScalar nd-array (long n)))

(defn get-in-scalar
  "Gets the leaf element at specified location,
   following the standard java specification as a
   nd-array (0d-array)"
  ^INDArray
  [^INDArray nd-array indexes]
  (.getScalar nd-array (->long-array indexes)))

(defn get-double
  "Gets the nth leaf element of the flattended nd-array
   (line first) as a double"
  ^double
  [^INDArray nd-array n]
  (.getDouble nd-array (long n)))

(defn get-in-double
  "Gets the leaf element at specified location,
   following the standard java specification as a
   double"
  ^double
  [^INDArray nd-array indexes]
  (.getDouble nd-array (->long-array indexes)))

;;=================Components===============

(defn get-row
  "Gets nth row of the given nd-array"
  [^INDArray nd-array n]
  (.getRow nd-array (long n)))

(defn get-column
  "Gets nth column of the given nd-array"
  [^INDArray nd-array n]
  (.getColumn nd-array (long n)))

;;////////////////////////////////////////////////////////////////////////////
;;============================================================================
;;                                 SUBSPACES
;;============================================================================
;;\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

;;===================================================
;;=======================CLONE=======================
;;===================================================

(defn clone
  "Clones a nd-array.
   Useful for immutability"
  ^INDArray
  [^INDArray nd-array]
  (.dup nd-array))

;;////////////////////////////////////////////////////////////////////////////
;;============================================================================
;;                              SELF-OPERATIONS
;;============================================================================
;;\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

;;=======================================================
;;=======================IMMUTABLE=======================
;;=======================================================

(defn roll-axis
  ^INDArray
  ([^INDArray nd-array index]
   (.rollAxis nd-array (long index)))
  ([^INDArray nd-array start end]
   (.rollAxis nd-array (long start) (long end))))

(defn transpose
  "NDArray transposition"
  ^INDArray
  [^INDArray nd-array]
  (.transpose nd-array))

(defn normalize
  "NDArray statistical normalization
   (zero mean and unit variance.
   Preserves input unlike normalize!"
  ^INDArray
  [^INDArray a]
  (Transforms/normalizeZeroMeanAndUnitVariance (clone a)))

(defn normalize!
  "NDArray statistical normalization
   (zero mean and unit variance.
   Alters input unlike normalize"
  ^INDArray
  [^INDArray a]
  (Transforms/normalizeZeroMeanAndUnitVariance a))

(defn arg-max
  "Returns a NDArray of maximal values along dimension"
  ^INDArray
  [^INDArray a dimension]
  (Nd4j/argMax a ^ints (->int-array dimension)))

(defn arg-min
  "Returns a NDArray of muinimal values along dimension"
  ^INDArray
  [^INDArray a dimension]
  (Nd4j/argMin a ^ints (->int-array dimension)))

(defn pad
  ^INDArray
  ([^INDArray a width]
   (Nd4j/pad a (->iint-array width)))
  ([^INDArray a width value]
   (pad a width :constant value))
  ([^INDArray a width mode value]
   (Nd4j/pad a (->iint-array width) (->pad-mode mode) (double value))))
   
(defn append
  ^INDArray
  [^INDArray a width value axis]
  (Nd4j/append a (int width) (double value) (int axis)))

(defn prepend
  ^INDArray
  [^INDArray a width value axis]
  (Nd4j/prepend a (int width) (double value) (int axis)))

;;=======================================================
;;=======================FUNCTIONS=======================
;;=======================================================

;;=================Meta===============

;; From https://stackoverflow.com/questions/50801908/is-it-possible-to-define-a-macroanything-else-to-dispatch-on-functions
(defmacro defoperator [fn-name static-name]
  `(defn ~fn-name
     ^INDArray
     [^INDArray a#]
     (. Transforms ~static-name a# true)))

(defmacro defoperator! [fn-name static-name]
  `(defn ~fn-name
     ^INDArray
     [^INDArray a#]
     (. Transforms ~static-name a# false)))

;;=================Negate===============

(defoperator negate neg)
(defoperator! negate! neg)

;;=================Abs===============

(defoperator abs abs)
(defoperator! abs! abs)

;;=================Floor===============

(defoperator floor floor)
(defoperator! floor! floor)

;;=================Ceil===============

(defoperator ceil ceiling)
(defoperator! ceil! ceiling)

;;=================Round===============

(defoperator round round)
(defoperator! round! round)

;;=================Stabilize===============

(defn stabilize
  ^INDArray
  [k ^INDArray a]
  (Transforms/stabilize a ^double (double k) true))

(defn stabilize!
  ^INDArray
  [k ^INDArray a]
  (Transforms/stabilize a ^double (double k) false))

;;=================Sqrt===============

(defoperator sqrt sqrt)
(defoperator! sqrt! sqrt)

;;=================Exp===============

(defoperator exp exp)
(defoperator! exp! exp)

;;=================Cosh===============

(defoperator cosh cosh)
(defoperator! cosh! cosh)

;;=================Sinh===============

(defoperator sinh sinh)
(defoperator! sinh! sinh)

;;=================Tanh===============

(defoperator tanh tanh)
(defoperator! tanh! tanh)

;;=================Hard tanh===============

(defoperator hard-tanh hardTanh)
(defoperator! hard-tanh! hardTanh)

;;=================Hard tanh derivative===============

(defoperator dhard-tanh hardTanhDerivative)
(defoperator! dhard-tanh! hardTanhDerivative)

;;=================Atanh===============

(defoperator atanh atanh)
(defoperator! atanh! atanh)

;;=================Ln===============

(defoperator ln log)
(defoperator! ln! log)

;;=================Log===============

(defn log
  ^INDArray
  [base ^INDArray a]
  (Transforms/log a ^double (double base) true))

(defn log!
  ^INDArray
  [base ^INDArray a]
  (Transforms/log a ^double (double base) false))

;;=================Sign===============

(defoperator sign sign)
(defoperator! sign! sign)

;;=================Cos===============

(defoperator cos cos)
(defoperator! cos! cos)

;;=================Sin===============

(defoperator sin sin)
(defoperator! sin! sin)

;;=================Acos===============

(defoperator acos acos)
(defoperator! acos! acos)

;;=================Asin===============

(defoperator asin asin)
(defoperator! asin! asin)

;;=================Atan===============

(defoperator atan atan)
(defoperator! atan! atan)

;;=================Sigmoid===============

(defoperator sigmoid sigmoid)
(defoperator! sigmoid! sigmoid)

;;=================Sigmoid derivative===============

(defoperator dsigmoid sigmoidDerivative)
(defoperator! dsigmoid! sigmoidDerivative)

;;=================Eps===============

;; WTF ?!
;; (defoperator eps eps)
;; (defoperator! eps! eps)

;;=================Softmax===============

(defoperator softmax softmax)
(defoperator! softmax! softmax)

;;=================Softplus===============

(defoperator softplus softPlus)
(defoperator! softplus! softPlus)

;;=================Softsign===============

(defoperator softsign softsign)
(defoperator! softsign! softsign)

;;=================Softsign derivative===============

(defoperator dsoftsign softsignDerivative)
(defoperator! dsoftsign! softsignDerivative)

;;=================Relu===============

(defoperator relu relu)
(defoperator! relu! relu)

;;=================Relu6===============

(defoperator relu6 relu6)
(defoperator! relu6! relu6)

;;=================Leaky relu===============

(defoperator leaky-relu leakyRelu)
(defoperator! leaky-relu! leakyRelu)

;;=================Leaky relu derivative===============

(defoperator dleaky-relu leakyReluDerivative)
(defoperator! dleaky-relu! leakyReluDerivative)

;;=================Elu===============

(defoperator elu elu)
(defoperator! elu! elu)

;;=================Elu derivative===============

(defoperator delu eluDerivative)
(defoperator! delu! eluDerivative)

;;=================Meta===============

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
;;   :eps eps
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

(def single-ops!
  {:negate negate!
   :abs abs!
   :floor floor!
   :ceil ceil!
   :round! round!
   :stabilize stabilize!
   :sqrt sqrt!
   :exp exp!
   :cosh cosh!
   :sinh sinh!
   :tanh tanh!
   :hard-tanh hard-tanh!
   :dhard-tanh dhard-tanh!
   :atanh atanh!
   :ln ln!
   :log log!
   :sign sign!
   :cos cos!
   :sin sin!
   :acos acos!
   :asin asin!
   :atan atan
   :sigmoid sigmoid!
   :dsigmoid dsigmoid!
;;   :eps eps!
   :softmax softmax!
   :softplus softplus!
   :softsign softsign!
   :dsoftsign dsoftsign!
   :relu relu!
   :relu6 relu6!
   :leaky-relu leaky-relu!
   :dleaky-relu dleaky-relu!
   :elu elu!
   :delu delu!})

;;=================Generic===============

(defn transform
  "Transforms a vector using an element-wise operator.
   This operation preserves input. See transform! for
   destructive operations.
   Slower than directly calling the specific operation,
   use this when needed.
   Usage (NDArray representation simplified for readibility, this is not a vector !) :
   (transform :ln [1.0 2.0 3.0])
   ~
   (ln [1.0 2.0 3.0])
   =
   [0.0 0.693.. 1.098..]"
  ^INDArray
  ([kind ^INDArray a]
   (if-let [f (kind single-ops)]
     (f a)
     (throw (Exception. (str "TRANSFORM - Unknown operator : " kind)))))
  ([kind arg1 ^INDArray a]
   (if-let [f (kind single-ops)]
     (f arg1 a)
     (throw (Exception. (str "TRANSFORM - Unknown operator : " kind)))))
  ([kind  arg1 arg2 ^INDArray a]
   (if-let [f (kind single-ops)]
     (f arg1 arg2 a)
     (throw (Exception. (str "TRANSFORM - Unknown operator : " kind))))))

(defn transform!
  "Transforms a vector using an element-wise operator.
   This is a destructive operation, i.e. the input is
   altered to target value. Use 'transform' for immutability.
   Slower than directly calling the specific operation,
   use this when needed.
   Usage (NDArray representation simplified for readibility, this is not a vector !) :
   (transform! :ln [1.0 2.0 3.0])
   ~
   (ln! [1.0 2.0 3.0])
   =
   [0.0 0.693.. 1.098..]"
  ^INDArray
  ([kind ^INDArray a]
   (if-let [f (kind single-ops!)]
     (f a)
     (throw (Exception. (str "TRANSFORM - Unknown operator : " kind)))))
  ([kind ^INDArray a arg1]
   (if-let [f (kind single-ops!)]
     (f a arg1)
     (throw (Exception. (str "TRANSFORM - Unknown operator : " kind)))))
  ([kind ^INDArray a arg1 arg2]
   (if-let [f (kind single-ops!)]
     (f a arg1 arg2)
     (throw (Exception. (str "TRANSFORM - Unknown operator : " kind))))))

;;////////////////////////////////////////////////////////////////////////////
;;============================================================================
;;                             BETWEEN-OPERATIONS
;;============================================================================
;;\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

(defn t*
  ([^INDArray left ^INDArray right axis]
   (Nd4j/tensorMmul left right (->iint-array axis))))


(defn n1-distance
  [^INDArray a1 ^INDArray a2]
  (.distance1 a1 a2))

(defn n2-distance
  [^INDArray a1 ^INDArray a2]
  (.distance2 a1 a2))

(defn cosine-distance
  [^INDArray a1 ^INDArray a2]
  (Transforms/cosineDistance a1 a2))

(defn manhattan-distance
  [^INDArray a1 ^INDArray a2]
  (Transforms/manhattanDistance a1 a2))

(defn jaccard-distance
  [^INDArray a1 ^INDArray a2]
  (Transforms/jaccardDistance a1 a2))

(defn hamming-distance
  [^INDArray a1 ^INDArray a2]
  (Transforms/hammingDistance a1 a2))

(defn distance
  "Computes distance between two
   NDArray instances using euclidean distance
   by default (two arities) or the selected
   method.
   Methods are : :abs (n1), :euclidean (n2),
                 :cos, :manhattan, :jaccard,
                 :hamming
   Usage :
   (distance nd1 nd2)
   (distance :method nd1 nd2)"
  ([a1 a2]
   (n2-distance a1 a2))
  ([kind a1 a2]
   (case kind
         :abs (n1-distance a1 a2)
         :euclidean (n2-distance a1 a2)
         :cos (cosine-distance a1 a2)
         :manhattan (manhattan-distance a1 a2)
         :jaccard (jaccard-distance a1 a2)
         :hamming (hamming-distance a1 a2))))

;;=======================================================
;;=======================FUNCTIONS=======================
;;=======================================================

;;=================Meta===============

;; From https://stackoverflow.com/questions/50801908/is-it-possible-to-define-a-macroanything-else-to-dispatch-on-functions
(defmacro defcooperator [fn-name static-name]
  `(defn ~fn-name
     ^INDArray
     [^INDArray n1# ^INDArray n2#]
     (. Transforms ~static-name n1# n2# true)))

(defmacro defcooperator! [fn-name static-name]
  `(defn ~fn-name
     ^INDArray
     [^INDArray n1# ^INDArray n2#]
     (. Transforms ~static-name n1# n2# false)))

;;=================Max===============

(defcooperator mmax max)
(defcooperator! mmax! max)

;;=================Pow===============

(defcooperator pow pow)
(defcooperator! pow! pow)

;;=================Meta===============

(def single-coops
  {:max mmax
   :pow pow})

(def single-coops!
  {:max mmax!
   :pow pow!})

;;=================Generic===============

(defn cotransform
  "Transforms a vector using an element-wise operator.
   This operation preserves input. See transform! for
   destructive operations.
   Slower than directly calling the specific operation,
   use this when needed.
   Usage (NDArray representation simplified for readibility, this is not a vector !) :
   (transform :ln [1.0 2.0 3.0])
   ~
   (ln [1.0 2.0 3.0])
   =
   [0.0 0.693.. 1.098..]"
  ^INDArray
  ([kind ^INDArray a1 ^INDArray a2]
   (if-let [f (kind single-coops)]
     (f a1 a2)
     (throw (Exception. (str "TRANSFORM - Unknown operator : " kind)))))
  ([kind arg1 ^INDArray a1 ^INDArray a2]
   (if-let [f (kind single-coops)]
     (f arg1 a1 a2)
     (throw (Exception. (str "TRANSFORM - Unknown operator : " kind)))))
  ([kind  arg1 arg2 ^INDArray a1 ^INDArray a2]
   (if-let [f (kind single-coops)]
     (f arg1 arg2 a1 a2)
     (throw (Exception. (str "TRANSFORM - Unknown operator : " kind))))))

(defn cotransform!
  "Transforms a vector using an element-wise operator.
   This is a destructive operation, i.e. the input is
   altered to target value. Use 'transform' for immutability.
   Slower than directly calling the specific operation,
   use this when needed.
   Usage (NDArray representation simplified for readibility, this is not a vector !) :
   (transform! :ln [1.0 2.0 3.0])
   ~
   (ln! [1.0 2.0 3.0])
   =
   [0.0 0.693.. 1.098..]"
  ^INDArray
  ([kind ^INDArray a1 ^INDArray a2]
   (if-let [f (kind single-coops!)]
     (f a1 a2)
     (throw (Exception. (str "TRANSFORM - Unknown operator : " kind)))))
  ([kind arg1 ^INDArray a1 ^INDArray a2]
   (if-let [f (kind single-coops!)]
     (f arg1 a1 a2)
     (throw (Exception. (str "TRANSFORM - Unknown operator : " kind)))))
  ([kind  arg1 arg2 ^INDArray a1 ^INDArray a2]
   (if-let [f (kind single-coops!)]
     (f arg1 arg2 a1 a2)
     (throw (Exception. (str "TRANSFORM - Unknown operator : " kind))))))

;;////////////////////////////////////////////////////////////////////////////
;;============================================================================
;;                                 CLOJURE
;;============================================================================
;;\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

;;======================================================
;;=======================INTERNAL=======================
;;======================================================

(defn- slice->clj
  "Internal usage.
   Recursive function which transforms
   a slice of a nd-array into its clojure
   representation (vector or double)"
  [^INDArray nd-array shape]
  (if (< 1 (count shape))
    (let [next-shape (rest shape)]
      (mapv
        (fn [i]
          (slice->clj (get-row nd-array i) next-shape))
        (range (first shape))))
    (mapv
      (fn [i]
        (get-double nd-array i))
      (range (first shape)))))

;;======================================================
;;=======================EXECUTOR=======================
;;======================================================

(defmethod ->clj INDArray
  [^INDArray nd-array]
  (let [shape (->clj (get-shape nd-array))]
    (slice->clj nd-array shape)))

;;////////////////////////////////////////////////////////////////////////////
;;============================================================================
;;                                  WRITERS
;;============================================================================
;;\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

(defn write-nd-array!
  "Writes a nd-array directly into a text file
   (.txt or .csv). The main advantage is to avoid
   the potentially costly conversion to clojure code.
   By default, the split is ';'"
  ([^String path ^INDArray nd-array]
   (Nd4j/writeTxt nd-array path ";"))
  ([^String path ^String split ^INDArray nd-array]
   (Nd4j/writeTxt nd-array path split)))
