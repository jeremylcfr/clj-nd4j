(ns repl
  (:require [clj-nd4j [ndarray :as nda]
                      [dataset :as ndd]]
            [clj-nd4j.dataset [normalization :as norm]]
            [clj-nd4j.ml [activation :as mlactiv]
                         [gradient :as mlgrad]
                         [learning :as mlearn]
                         [loss :as mloss]]
            [clj-java-commons [core :refer :all]
                              [coerce :refer [->clj]]])
  (:import [org.nd4j.linalg.factory Nd4j])
  (:refer-clojure :exclude [/ neg? pos?]))

;; https://github.com/deeplearning4j/deeplearning4j/tree/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/dataset/api/preprocessor