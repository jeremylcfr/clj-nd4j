(ns clj-nd4j.ndarray.matrix
  (:require [clj-nd4j.ndarray :as core]
            [clj-java-commons.core :refer :all]
            [clj-java-commons.coerce :refer [->clj]])
  (:import [org.nd4j.linalg.factory Nd4j]
           [org.nd4j.linalg.api.ndarray INDArray]
           [org.nd4j.linalg.ops.transforms Transforms])
  (:refer-clojure :exclude [/ pos? neg?]))

;; EXPERIMENTAL : subset of core limited to matrices
;; Matrix is a specific case of NDArray...a [n m] array

;;////////////////////////////////////////////////////////////////////////////
;;============================================================================
;;                                  BUILDERS
;;============================================================================
;;\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

;;=====================================================
;;=======================GENERIC=======================
;;=====================================================

;;=================Internal===============

;;=================Executor===============

(defn ->matrix
  "Creates a matrix from arguments.
   The single-arity version is a native method
   contrary to the general ->nd-array builder
   and should be favoured unless you know
   what you are doing (2-arity version does not
   offer any guarantee about output type because
   this is equivalent to ->nd-array)"
  ^INDArray
  ([data]
   (Nd4j/create (->ddouble-array data)))
  ([shape data]
   (core/->nd-array shape data)))

(defn ->safe-matrix
  "Same as ->matrix but performs a pre-check on arguments
   to avoid calling the function. This version is more
   useful for the 2-arity builder since it can succeed while
   not returning a matrix.
   This operation involves an increased cost at creation time
   but can be useful from a debugging viewpoint"
  ^INDArray
  ([data]
   (let [first-item (first data)]
     (if (and (sequential? data) (sequential? first-item) (number? (first first-item)))
       (->matrix data)
       (throw (Exception. (str "Matrix creation error : " data " is not a valid matrix representation"))))))
  ([shape data]
   (if (= 2 (count shape))
     (try (core/->nd-array shape data)
       (catch Exception e (throw (Exception. (str "Matrix creation error : " data " is not a matrix OR shape " shape " is not valid")))))
     (throw (Exception. (str "Shape : " shape " is not a matrix shape"))))))

(defn mpow
  ^INDArray
  [n ^INDArray m]
  (Transforms/mpow m (int n) true))

(defn mpow!
  ^INDArray
  [n ^INDArray m]
  (Transforms/mpow m (int n) false))

(defn m*
  "Matrix multiplication.
   Input style 1 (no transposition) :
   - left : left matrix
   - right : right matrix
   Input style 2 (transposition) :
   - left : left matrix
   - right : right matrix
   - transpose-left : when true, transpose matrix left before multiplication
   - transpose-left : when true, transpose matrix right before multiplication"
  ^INDArray
  ([^INDArray left ^INDArray right]
   (Nd4j/gemm left right false false))
  ([^INDArray left ^INDArray right transpose-left transpose-right]
   (Nd4j/gemm left right (boolean transpose-left) (boolean transpose-right))))

(defn nd-matrix?
  [^INDArray obj]
  (.isMatrix obj))

(defn matrix?
  [obj]
  (and (core/nd-array? obj) (nd-matrix? obj)))
