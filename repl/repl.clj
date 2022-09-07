(ns repl
  (:require [clj-nd4j [ndarray :as nda]
                      [dataset :as ndd]]
            [clj-java-commons [core :refer :all]
                              [coerce :refer [->clj]]])
  (:import [org.nd4j.linalg.factory Nd4j])
  (:refer-clojure :exclude [/ neg? pos?]))