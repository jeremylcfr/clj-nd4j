(ns clj-nd4j.dataset
  (:require [clj-nd4j.ndarray :as core]
            [clj-java-commons.core :refer :all]
            [clj-java-commons.coerce :refer [->clj]]
            [clojure.algo.generic.functor :refer [fmap]]
            [clojure.java.io :as io])
  (:import [org.nd4j.linalg.api.ndarray INDArray]
           [org.nd4j.linalg.dataset DataSet SplitTestAndTrain])
  (:refer-clojure :exclude [/ pos? neg? shuffle]))

;;////////////////////////////////////////////////////////////////////////////
;;============================================================================
;;                                 PREDICATES
;;============================================================================
;;\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

(defn dataset?
  [obj]
  (instance? DataSet obj))

;;////////////////////////////////////////////////////////////////////////////
;;============================================================================
;;                                  BUILDERS
;;============================================================================
;;\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

;;=================================================
;;=======================RAW=======================
;;=================================================

(defn dataset
  "Strict wrapper of DataSet constructors, i.e.
   accepts only NDArray instances as input.
   Use ->dataset for a more flexible (but
   a bit less performant) coercer"
  ^DataSet
  ([]
   (DataSet.))
  ([^INDArray features ^INDArray labels]
   (DataSet. features labels))
  ([^INDArray features ^INDArray labels ^INDArray features-mask ^INDArray labels-mask]
   (DataSet. features labels features-mask labels-mask)))

;;=====================================================
;;=======================CLOJURE=======================
;;=====================================================

;;=================Abstraction===============

(defmulti clj->dataset
  (fn [obj]
    (cond (map? obj)
            :map
          (sequential? obj)
            :seq
          (nil? obj)
            :nil
          :else
            :error)))

;;=================Implementation===============

(defmethod clj->dataset :map
  [{:keys [features labels features-mask labels-mask]}]
  ^DataSet
  (if (and features labels)
    (if (or features-mask labels-mask)
      (DataSet. ^INDArray (core/->nd-array features) ^INDArray (core/->nd-array labels) ^INDArray (core/->nilable-nd-array features-mask) ^INDArray (core/->nilable-nd-array labels-mask))
      (DataSet. ^INDArray (core/->nd-array features) ^INDArray (core/->nd-array labels)))
    (DataSet.)))

(defmethod clj->dataset :seq
  [obj]
  ^DataSet
  (case (count obj)
        2  (DataSet. ^INDArray (core/->nd-array (obj 0)) ^INDArray (core/->nd-array (obj 1)))
        4  (DataSet. ^INDArray (core/->nd-array (obj 0)) ^INDArray (core/->nd-array (obj 1)) ^INDArray (core/->nilable-nd-array (obj 2)) ^INDArray (core/->nilable-nd-array (obj 3)))
        0  (DataSet.)
        (throw (Exception. "DATASET - DataSet can only be created from sequential objects of size 0, 2 or 4"))))

(defmethod clj->dataset :nil
  [obj]
  ^DataSet
  (DataSet.))

(defmethod clj->dataset :error
  [obj]
  ^DataSet
  (throw (Exception. (str "DATASET - Input " obj " is not coercible to dataset, see documentation"))))

;;=====================================================
;;=======================GENERIC=======================
;;=====================================================

(defn ->dataset
  "Generic dataset coercer. Uses ->nd-array
   to coerce input which can be so a clojure structure.
   Single-arity version does nothing if input is a DataSet
   and parses a clojure structure if valid (note that it is the
   less performant way of doing this)
   You can pass a map :
   {:features ... :labels ... :features-mask ... :labels-mask ...}
   or a nested/empty vector (in same order)
   [[ ... ] [ ... ] [ ... ] [ ... ]]
   or nil (~ no arg)
   Valid args are either :
   - no arg
   - two args (features and labels)
   - four args (all but masks are nilable)"
  ^DataSet
  ([]
   (DataSet.))
  ([obj]
   (if (dataset? obj)
     obj
     (clj->dataset obj)))
  ([features labels]
   (DataSet. ^INDArray (core/->nd-array features) ^INDArray (core/->nd-array labels)))
  ([features labels features-mask labels-mask]
   (DataSet. ^INDArray (core/->nd-array features) ^INDArray (core/->nd-array labels) ^INDArray (core/->nilable-nd-array features-mask) ^INDArray (core/->nilable-nd-array labels-mask))))

;;=======================================================
;;=======================FROM FILE=======================
;;=======================================================

(defn load-dataset
  "Loads a dataset from a file written preferably with write-dataset!"
  ^DataSet
  [io-coercible]
  (.load ^java.io.File (io/file io-coercible)))

;;////////////////////////////////////////////////////////////////////////////
;;============================================================================
;;                                  OPERATIONS
;;============================================================================
;;\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

;;======================================================
;;=======================MUTATION=======================
;;======================================================

(defn do-shuffle!
  [^DataSet data]
  (.shuffle data))

(defn shuffle!
  [^DataSet data]
  (do-shuffle! data)
  data)

(defn do-binarize!
  ([^DataSet data]
   (.binarize data))
  ([^DataSet data cutoff]
   (.binarize data ^double (double cutoff))))

(defn binarize!
  ^DataSet
  ([^DataSet data]
   (do-binarize! data)
   data)
  ([^DataSet data cutoff]
   (do-binarize! data cutoff)
   data))

(defn do-normalize!
  [^DataSet data]
  (.normalize data))

(defn normalize!
  ^DataSet
  [^DataSet data]
  (do-normalize! data)
  data)

(defn do-scale!
  [^DataSet data]
  (.scale data))

(defn scale!
  ^DataSet
  [^DataSet data]
  (do-scale! data)
  data)

(defn do-multiply!
  [^DataSet data multiplicator]
  (.multiplyBy data ^double (double multiplicator)))

(defn multiply!
  ^DataSet
  [^DataSet data multiplicator]
  (do-multiply! data multiplicator)
  data)

;;=======================================================
;;=======================IMMUTABLE=======================
;;=======================================================

(defn copy
  ^DataSet
  [^DataSet data]
  (.copy data))

(defn ->immutable-transformation1
  [transform-fn]
  (comp transform-fn copy))

(defn ->immutable-transformation2
  [transform-fn]
  (fn ^DataSet [^DataSet data arg]
    (-> (copy data)
        (transform-fn arg))))

(def shuffle
  (->immutable-transformation1 shuffle!))

(defn binarize
  ([^DataSet data]
   (-> (copy data)
       (binarize!)))
  ([^DataSet data cutoff]
   (-> (copy data)
       (binarize! cutoff))))

(def normalize
  (->immutable-transformation1 normalize!))

(def scale
  (->immutable-transformation1 scale!))

(def multiply
  (->immutable-transformation2 multiply!))

;;////////////////////////////////////////////////////////////////////////////
;;============================================================================
;;                                  ML
;;============================================================================
;;\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

(defn raw-split-test-and-train
  ^SplitTestAndTrain
  ([dataset num-samples]
   (raw-split-test-and-train dataset num-samples false))
  ([^DataSet dataset num-samples seed-or-random]
    (let [random (cond (instance? java.util.Random seed-or-random)
                         seed-or-random
                       (number? seed-or-random)
                         (java.util.Random. seed-or-random)
                       (true? seed-or-random)
                         (java.util.Random.)
                       (false? seed-or-random)
                         nil)]
      (if-not random
        (. dataset splitTestAndTrain ^int (int num-samples))
        (. dataset splitTestAndTrain ^int (int num-samples) ^java.util.Random random)))))


(defn split-test-and-train
  ([dataset num-samples]
   (split-test-and-train dataset num-samples false))
  ([^DataSet dataset num-samples seed-or-random]
    (let [splitter (raw-split-test-and-train dataset num-samples seed-or-random)
          train (.getTrain ^SplitTestAndTrain splitter)
          test (.getTest ^SplitTestAndTrain splitter)]
      {:train train
       :test test})))

(defn extract-features-and-labels
  [^DataSet dataset]
  {:labels (.getLabels dataset)
   :features (.getFeatures dataset)})

(defn deep-split-test-and-train
  ([dataset num-samples]
   (deep-split-test-and-train dataset num-samples false))
  ([dataset num-samples random]
   (->> (split-test-and-train dataset num-samples random)
        (fmap extract-features-and-labels))))


;;////////////////////////////////////////////////////////////////////////////
;;============================================================================
;;                                   I/O
;;============================================================================
;;\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

;;====================================================
;;=======================READER=======================
;;====================================================

(defn read-dataset!
  ^DataSet
  ([io-coercible]
   (read-dataset! io-coercible (dataset)))
  ([io-coercible ^DataSet data]
   (.load data ^java.io.File (io/file io-coercible))
   data))

(defn read-dataset
  ^DataSet
  ([io-coercible]
   (read-dataset! io-coercible))
  ([io-coercible ^DataSet data]
   (read-dataset! io-coercible (copy data))))

;;====================================================
;;=======================WRITER=======================
;;====================================================

(defn write-dataset!
  "Writes a dataset to a file which can be read
   back with load-dataset! saving some clojure interop
   loss"
  [io-coercible ^DataSet data]
  (.save data ^java.io.File (io/file io-coercible)))
