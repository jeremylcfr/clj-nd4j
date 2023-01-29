(ns clj-nd4j.ml.updaters
  (:require [clj-nd4j.ndarray :as nda]
            [clojure.algo.generic.functor :refer [fmap]]
            [clj-java-commons [core :refer :all]])
  (:import [org.nd4j.linalg.learning.config Nesterovs IUpdater]
           [org.nd4j.linalg.learning GradientUpdater NesterovsUpdater]
           [org.nd4j.linalg.schedule ISchedule]
           [java.util Map])
  (:refer-clojure :exclude [/ pos? neg? abs]))

(defn ->nesterovs-config
  ^Nesterovs
  ([options]
   (->nesterovs-config 
    {:learning-rate Nesterovs/DEFAULT_NESTEROV_LEARNING_RATE
     :momentum Nesterovs/DEFAULT_NESTEROV_MOMENTUM} options))
  ([defaults options]
   (let [{:keys [learning-rate momentum 
                 learning-rate-schedule
                 momentum-schedule]} (merge defaults options)]
     (cond (and learning-rate-schedule momentum-schedule)
             (Nesterovs. ^ISchedule learning-rate-schedule ^ISchedule momentum-schedule)
           learning-rate-schedule
            (Nesterovs. ^ISchedule learning-rate-schedule ^double (double momentum))
          momentum-schedule
            (Nesterovs. ^double (double learning-rate) ^ISchedule momentum-schedule)
          :else
            (Nesterovs. ^double (double learning-rate) ^double (double momentum))))))


(def updaters-config
  {:nesterovs {:builder ->nesterovs-config
               :defaults {:learning-rate Nesterovs/DEFAULT_NESTEROV_LEARNING_RATE
                          :momentum Nesterovs/DEFAULT_NESTEROV_MOMENTUM}}})

(defn ->updater-config
  ^IUpdater
  ([type-fn]
   (->updater-config type-fn nil))
  ([type-fn options]
   (let [{:keys [builder defaults]} (get updaters-config type-fn)]
     (builder defaults options))))

(defn iupdater?
  [obj]
  (instance? IUpdater obj))

(defn ->nesterovs
  ^NesterovsUpdater
  ([]
   (->nesterovs nil))
  ([config]
   (NesterovsUpdater. ^Nesterovs (cond-> config (not (iupdater? config)) (->nesterovs-config)))))

(def updaters
  {:nesterovs ->nesterovs})

(defn ->updater
  ^GradientUpdater
  ([type-fn]
   (->updater type-fn nil))
  ([type-fn config]
   (let [builder (get updaters type-fn)]
     (builder (->updater-config type-fn config)))))

(defn get-config
  ^IUpdater
  [^GradientUpdater obj]
  (.getConfig obj))
    
(defn get-state
  ^Map
  [^GradientUpdater obj]
  (.getState obj))
    
(defn set-state!
  [^GradientUpdater obj ^Map state-map initialize?]
  (.setState obj ^Map (fmap nda/->nd-array state-map) ^boolean (boolean initialize?)))

(defn set-state-view-array!
  [^GradientUpdater obj gradient-shape gradient-order initialize?]
  (.setStateViewArray obj (->long-array gradient-shape) (char gradient-order) ^boolean (boolean initialize?)))

(defn apply-updater!
  [^GradientUpdater obj gradient iteration epoch]
  (.applyUpdater obj (nda/->nd-array gradient) ^int (int iteration) ^int (int epoch)))
    

    

  

