(ns clj-nd4j.dataset.normalization
  (:require [clj-nd4j.ndarray :as nd4j])
  (:import [org.nd4j.linalg.dataset.api.preprocessor Normalizer AbstractDataSetNormalizer NormalizerMinMaxScaler NormalizerStandardize]
           [org.nd4j.linalg.dataset.api.preprocessor.stats DistributionStats]
           [org.nd4j.linalg.api.ndarray INDArray]
           [org.nd4j.linalg.dataset.api DataSet]
           [org.nd4j.linalg.dataset.api.iterator DataSetIterator]))

(def normalizer-options-paths
  {:min-max-scaler [:dataset]
   :standardizer   [:dataset]})

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
    (.setLabelStats ^NormalizerMinMaxScaler normalizer ^INDArray (nd4j/->nd-array label-min)   ^INDArray (nd4j/->nd-array label-max))))

(defmethod apply-normalizer-options* :standardizer
  [_ {:keys [feature-mean feature-std label-mean label-std]} ^NormalizerStandardize normalizer]
  ;; doto-cond, see where to use it
  ;; see how to handle XOR case (incorrect) ==> clojure.spec ?
  (when (or feature-mean feature-std)
    (.setFeatureStats ^NormalizerStandardize normalizer (DistributionStats. ^INDArray (nd4j/->nd-array feature-mean) ^INDArray (nd4j/->nd-array feature-std))))
  (when (or label-mean label-std)
    (.setLabelStats ^NormalizerStandardize normalizer  ^INDArray (nd4j/->nd-array label-mean)   ^INDArray (nd4j/->nd-array label-std))))

(def apply-normalizer-options apply-normalizer-options*)

(defn apply-normalizer-options-stack
  [key-fn options normalizer]
  (let [stack (conj (get normalizer-options-paths key-fn) key-fn)]
    (doseq [normalizer-fn stack]
      (apply-normalizer-options normalizer-fn options normalizer))
    normalizer))



;; Not consistent, see later
(defn normalizer-min-max-scaler
  ^NormalizerMinMaxScaler
  ([{:keys [min-range max-range] :or {min-range 0.0 , max-range 1.0}}]
   (NormalizerMinMaxScaler. ^double (double min-range) ^double (double max-range)))
  ([builder-spec options]
   (let [normalizer (normalizer-min-max-scaler builder-spec)]
     (apply-normalizer-options-stack :min-max-scaler options normalizer)))
  ([options min-range max-range]
   (normalizer-min-max-scaler {:min-range min-range , :max-range max-range} options)))

(defn normalizer-standardizer
  ^NormalizerStandardize
  ([]
   (NormalizerStandardize.))
  ([options]
   (let [normalizer (normalizer-standardizer)]
     (apply-normalizer-options-stack :standardizer options normalizer))))




;; see later if it is the best granularity
(defn dataset-normalizer?
  [obj]
  (instance? AbstractDataSetNormalizer obj))

;; Add with iterator

(defn fit-dataset!
  [^AbstractDataSetNormalizer normalizer ^DataSet dataset]
  (.fit normalizer dataset))

(defn fit-dataset
  ^AbstractDataSetNormalizer
  [^AbstractDataSetNormalizer normalizer dataset]
  (fit-dataset! normalizer dataset)
  normalizer)

 (defn qualify-normalizer
   [normalizer]
   (cond (dataset-normalizer? normalizer) :single))

 (def fitters
   {:single {:void fit-dataset! , :self fit-dataset}})


 (defn fit!
   ([normalizer dataset]
    (fit! (qualify-normalizer normalizer) normalizer dataset))
   ([type-fn normalizer dataset]
    (let [fitter (get-in fitters [type-fn :void])]
      (fitter normalizer dataset))))

 (defn fit
   ^Normalizer
   ([normalizer dataset]
    (fit! normalizer dataset)
    normalizer)
   ([type-fn normalizer dataset]
    (fit! type-fn normalizer dataset)
    normalizer))

 (defn transform!
   [^AbstractDataSetNormalizer normalizer ^DataSet obj]
   (.transform normalizer obj))

 (defn transform
   ^DataSet
   [normalizer obj]
   (transform! normalizer obj)
    obj)

;; labels and features

 (defn revert!
   [^AbstractDataSetNormalizer normalizer ^DataSet dataset]
   (.revert normalizer dataset))

 (defn revert
   ^DataSet
   [normalizer dataset]
   (revert! normalizer dataset)
   dataset)

 (defn revert-features!
   ([^AbstractDataSetNormalizer normalizer ^INDArray features]
    (.revertFeatures normalizer features))
   ([^AbstractDataSetNormalizer normalizer ^INDArray features ^INDArray features-mask]
    (.revertFeatures normalizer features features-mask)))

 (defn revert-features
   ^INDArray
   ([normalizer features]
    (revert-features! normalizer features)
    features)
   ([normalizer features features-mask]
    (revert-features! normalizer features features-mask)
    features))

(defn revert-labels!
  ([^AbstractDataSetNormalizer normalizer ^INDArray labels]
   (.revertLabels normalizer labels))
  ([^AbstractDataSetNormalizer normalizer ^INDArray labels ^INDArray labels-mask]
   (.revertLabels normalizer labels labels-mask)))

(defn revert-labels
  ^INDArray
  ([normalizer labels]
   (revert-labels! normalizer labels)
   labels)
  ([normalizer labels labels-mask]
   (revert-labels! normalizer labels labels-mask)
   labels))
