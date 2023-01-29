(ns clj-nd4j.ml.learning
  (:import [org.nd4j.linalg.learning.config IUpdater Nesterovs]))

;; Schedule à ajouter
;; Que learning-rate
(defn ->nesterovs-config
  ^Nesterovs
  [{:keys [momentum learning-rate] :or {momentum Nesterovs/DEFAULT_NESTEROV_MOMENTUM , learning-rate Nesterovs/DEFAULT_NESTEROV_LEARNING_RATE}}]
  (Nesterovs. ^double (double learning-rate) ^double (double momentum)))
        
(def updater-configs
  {:nesterovs ->nesterovs-config})

(defn ->updater-config
  ([opts]
   (->updater-config (:type opts) opts))
  ([kind opts]
   (let [builder (get updater-configs kind)]
     (builder opts))))

;; Objet à partir de config ? Utile ?