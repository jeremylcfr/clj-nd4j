(ns clj-nd4j.ml.evaluation
  (:require [clj-nd4j.ndarray :as nda])
  (:import [org.nd4j.evaluation IEvaluation]
           [org.nd4j.evaluation.regression Evaluation RegressionEvaluation]
           [org.nd4j.linalg.api.ndarray INDArray]
           [java.util List])
  (:refer-clojure :exclude [merge reset!]))

 public Evaluation ()

 public Evaluation (int numClasses)
 public Evaluation (List<String> labels)
 public Evaluation (Map<Integer, String> labels)
 public Evaluation (double binaryDecisionThreshold)
 public Evaluation (INDArray costArray)

 public Evaluation (int numClasses, Integer binaryPositiveClass)
 public Evaluation (List<String> labels, int topN)
 public Evaluation (double binaryDecisionThreshold, @NonNull Integer binaryPositiveClass)
 public Evaluation (List<String> labels, INDArray costArray)

 ;; Use clojure.spec ...
 (defn evaluation
   ^Evaluation
   ([]
    (Evaluation.))
   ([{:keys [num-classes labels binary-decision-thresold cost-array binary-positive-class top-n]}]
    (cond num-classes
            (if (or labels binary-decision-thresold cost-array)
              (throw (Exception. "Evaluation cannot be built with num-classes and provided argments"))
              (if binary-positive-class 
                (Evaluation. ^int (int num-classes) ^Integer binary-positive-class)
                (Evaluation. ^int (int num-classes))))
          labels
            (if (or (or num-classes binary-decision-thresold) (and top-n cost-array))
              (throw (Exception. "Evaluation cannot be built with labels and provided argments"))
              (cond top-n
                      (Evaluation. ^List (seq labels) ^int (int top-n))
                    cost-array
                      (Evaluation. ^List (seq labels) ^INDArray (nda/->nd-array cost-array))
                    (map? labels)
                      (Evaluation. ^Map labels)
                    :else
                      (Evaluation. ^List (seq labels))))
          ;; continuer
    )))
                      

(defn regression-evaluation
  ^RegressionEvaluation
  ([]
   (RegressionEvaluation.))
  ([{:keys [num-cols cols-names precision] :or {precision RegressionEvaluation/DEFAULT_PRECISION}}]
   (cond cols-names
           (RegressionEvaluation. ^List (seq cols-names) ^long (long precision))
         num-cols
           (RegressionEvaluation. ^long (long num-cols) ^long (long precision))
         :else
           (regression-evaluation))))

(defn regression-evaluation?
  [obj]
  (instance? RegressionEvaluation obj))

(defn ->regression-evaluation
  ^RegressionEvaluation
  ([]
   (regression-evaluation))
  ([obj]
   (if (regression-evaluation? obj)
     obj
     (regression-evaluation obj))))

;; see later for recordMetaData
(defn evaluate!
  ([^IEvaluation evaluation labels predictions]
   (.eval evaluation ^INDArray (nda/->nd-array labels) ^INDArray (nda/->nd-array predictions)))
  ([^IEvaluation evaluation labels predictions mask]
   (.eval evaluation ^INDArray (nda/->nd-array labels) ^INDArray (nda/->nd-array predictions) ^INDArray (nda/->nd-array mask))))

(defn evaluate
  ^IEvaluation
  ([evaluation labels predictions]
   (evaluate! evaluation labels predictions)
   evaluation)
  ([evaluation labels predictions mask]
   (evaluate! evaluation labels predictions mask)
   evaluation))
     
(defn merge!
  [^IEvaluation base-evaluation ^IEvaluation merged-evaluation]
  (.merge base-evaluation merged-evaluation))
    
(defn merge
  ^IEvaluation
  [base-evaluation merged-evaluation]
  (merge! base-evaluation merged-evaluation)
  base-evaluation)
    
(defn reset!
  [^IEvaluation evaluation]
  (.reset evaluation))
     
(defn reset
  ^IEvaluation
  [evaluation]
  (reset! evaluation)
  evaluation)

(defn ->string
  ^String
  [^IEvaluation evaluation]
  (.stats evaluation))

(defn ->json
  ^String
  [^IEvaluation evaluation]
  (.toJson evaluation))

(defn ->yaml
  ^String
  [^IEvaluation evaluation]
  (.toYaml evaluation))

;; get-value see how to do

;; better name ?
(defn blank-clone
  ^IEvaluation
  [^IEvaluation evaluation]
  (.newInstance evaluation))
    
     
