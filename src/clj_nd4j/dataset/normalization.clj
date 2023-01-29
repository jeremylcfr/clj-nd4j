(ns clj-nd4j.dataset.normalization
  (:require [clj-nd4j.ndarray :as nd4j])
  (:import [org.nd4j.linalg.dataset.api.preprocessor Normalizer AbstractDataSetNormalizer NormalizerMinMaxScaler]
           [org.nd4j.linalg.dataset.api.iterator DataSetIterator]
           [org.nd4j.linalg.api.ndarray INDArray]))

(def normalizer-options-paths
  {:min-max-scaler [:dataset]})

(defmulti apply-normalizer-options*
  (fn [normalizer-fn _ _] normalizer-fn))

(defmethod apply-normalizer-options* :dataset
  [_ {:keys [fit-label?]} ^AbstractDataSetNormalizer normalizer]
  ;; doto-cond, see where to use it
  (when (not (nil? fit-label?))
    (.fitLabel ^AbstractDataSetNormalizer normalizer ^boolean fit-label?)))

(defmethod apply-normalizer-options* :min-max-scaler
  [_ {:keys [feature-min feature-max label-min label-max]} ^NormalizerMinMaxScaler normalizer]
  ;; doto-cond, see where to use it
  ;; see how to handle XOR case (incorrect) ==> clojure.spec ?
  (when (or feature-min feature-max)
    (.setFeatureStats ^NormalizerMinMaxScaler normalizer ^INDArray (nd4j/->nd-array feature-min) ^INDArray (nd4j/->nd-array feature-max)))
  (when (or label-min label-max)
    (.setLabelStats   ^INDArray (nd4j/->nd-array label-min)   ^INDArray (nd4j/->nd-array label-max))))

(def apply-normalizer-options apply-normalizer-options*)

(defn apply-normalizer-options-stack
  [key-fn options normalizer]
  (let [stack (conj (get normalizer-options-paths key-fn) key-fn)]
    (doseq [normalizer-fn stack]
      (apply-normalizer-options normalizer-fn options normalizer))
    normalizer))


(defn normalizer-min-max-scaler
  ^NormalizerMinMaxScaler
  ([{:keys [min-range max-range] :or {min-range 0.0 , max-range 1.0}}]
   (NormalizerMinMaxScaler. ^double (double min-range) ^double (double max-range)))
  ([builder-spec options]
   (let [normalizer (normalizer-min-max-scaler builder-spec)]
     (apply-normalizer-options-stack :min-max-scaler options normalizer)))
  ([options min-range max-range]
   (normalizer-min-max-scaler {:min-range min-range , :max-range max-range} options)))

;; see later if it is the best granularity
 (defn dataset-normalizer?
   [obj]
   (instance? AbstractDataSetNormalizer obj))

(defn fit-dataset!
  [^AbstractDataSetNormalizer normalizer ^DataSetIterator iterator]
  (.fit normalizer iterator))

(defn fit-dataset
  ^AbstractDataSetNormalizer
  [^AbstractDataSetNormalizer normalizer iterator]
  (fit-dataset! normalizer iterator)
  normalizer)
 
 (defn qualify-normalizer
   [normalizer]
   (cond (dataset-normalizer? normalizer) :single))
 
 (def fitters
   {:single {:void fit-dataset! , :self fit-dataset}})
 
 
 (defn fit!
   ([normalizer iterator]
    (fit! (qualify-normalizer normalizer) normalizer iterator))
   ([type-fn normalizer iterator]
    (let [fitter (get-in fitters [type-fn :void])]
      (fitter normalizer iterator))))

 (defn fit
   ^Normalizer
   ([normalizer iterator]
    (fit! normalizer iterator)
    normalizer)
   ([type-fn normalizer iterator]
    (fit! type-fn normalizer iterator)
    normalizer))
 
 ;; next : transform and pre-process
  
   