(ns clj-nd4j.ml.gradient
  (:require [clj-nd4j.ndarray :as nda]
            [clojure.algo.generic.functor :refer [fmap]]
            [clj-java-commons [core :refer :all]])
  (:import [org.nd4j.linalg.learning GradientUpdater AdaDeltaUpdater AMSGradUpdater AdaGradUpdater AdaMaxUpdater AdamUpdater NadamUpdater NesterovsUpdater RmsPropUpdater SgdUpdater]
           [org.nd4j.linalg.learning.config IUpdater AMSGrad AdaDelta AdaGrad AdaMax Adam Nadam Nesterovs RmsProp Sgd]
           [java.util Map])
  (:refer-clojure :exclude [/ pos? neg?]))

;;////////////////////////////////////////////////////////////////////////////
;;============================================================================
;;                                CONFIGURATION
;;============================================================================
;;\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

;;======================================================
;;=======================SPECIFIC=======================
;;======================================================

;;=================AMS grad===============

(defn ->ams-grad-config
  ^AMSGrad
  [{:keys [learning-rate beta1 beta2 epsilon]
    :or {learning-rate AMSGrad/DEFAULT_AMSGRAD_LEARNING_RATE ,
         beta1 AMSGrad/DEFAULT_AMSGRAD_BETA1_MEAN_DECAY ,
         beta2 AMSGrad/DEFAULT_AMSGRAD_BETA2_VAR_DECAY ,
         epsilon AMSGrad/DEFAULT_AMSGRAD_EPSILON}}]
  (AMSGrad. ^double (double learning-rate) ^double (double beta1) ^double (double beta2) ^double (double epsilon)))

;;=================Ada delta===============

(defn ->ada-delta-config
  ^AdaDelta
  [{:keys [rho epsilon] :or {rho AdaDelta/DEFAULT_ADADELTA_RHO , epsilon AdaDelta/DEFAULT_ADADELTA_EPSILON}}]
  (AdaDelta. ^double (double rho) ^double (double epsilon)))

;;=================Ada grad===============

(defn ->ada-grad-config
  ^AdaGrad
  [{:keys [learning-rate epsilon] :or {learning-rate AdaGrad/DEFAULT_ADAGRAD_LEARNING_RATE , epsilon AdaGrad/DEFAULT_ADAGRAD_EPSILON}}]
  (AdaGrad. ^double (double learning-rate) ^double (double epsilon)))

;;=================Ada max===============

(defn ->ada-max-config
  ^AdaMax
  [{:keys [learning-rate beta1 beta2 epsilon]
    :or {learning-rate AdaMax/DEFAULT_ADAMAX_LEARNING_RATE ,
         beta1 AdaMax/DEFAULT_ADAMAX_BETA1_MEAN_DECAY ,
         beta2 AdaMax/DEFAULT_ADAMAX_BETA2_VAR_DECAY,
         epsilon AdaMax/DEFAULT_ADAMAX_EPSILON}}]
  (AdaMax. ^double (double learning-rate) ^double (double beta1) ^double (double beta2) ^double (double epsilon)))

;;=================Adam===============

(defn ->adam-config
  ^Adam
  [{:keys [learning-rate beta1 beta2 epsilon]
    :or {learning-rate Adam/DEFAULT_ADAM_LEARNING_RATE ,
         beta1 Adam/DEFAULT_ADAM_BETA1_MEAN_DECAY ,
         beta2 Adam/DEFAULT_ADAM_BETA2_VAR_DECAY ,
         epsilon Adam/DEFAULT_ADAM_EPSILON}}]
  (Adam. ^double (double learning-rate) ^double (double beta1) ^double (double beta2) ^double (double epsilon)))

;;=================Nadam===============

(defn ->nadam-config
  ^Nadam
  [{:keys [learning-rate beta1 beta2 epsilon]
    :or {learning-rate Nadam/DEFAULT_NADAM_LEARNING_RATE ,
         beta1 Nadam/DEFAULT_NADAM_BETA1_MEAN_DECAY ,
         beta2 Nadam/DEFAULT_NADAM_BETA2_VAR_DECAY ,
         epsilon Nadam/DEFAULT_NADAM_EPSILON}}]
  (Nadam. ^double (double learning-rate) ^double (double beta1) ^double (double beta2) ^double (double epsilon)))

;;=================Nesterovs===============

(defn ->nesterovs-config
  ^Nesterovs
  [{:keys [learning-rate momentum] :or {learning-rate Nesterovs/DEFAULT_NESTEROV_LEARNING_RATE , momentum Nesterovs/DEFAULT_NESTEROV_MOMENTUM}}]
  (Nesterovs. ^double (double learning-rate) ^double (double momentum)))

;;=================RMS Prop===============

(defn ->rms-prop-config
  ^RmsProp
  [{:keys [learning-rate rms-decay epsilon]
    :or {learning-rate RmsProp/DEFAULT_RMSPROP_LEARNING_RATE ,
         rms-decay RmsProp/DEFAULT_RMSPROP_RMSDECAY ,
         epsilon RmsProp/DEFAULT_RMSPROP_EPSILON}}]
  (RmsProp. ^double (double learning-rate) ^double (double rms-decay) ^double (double epsilon)))

;;=================RMS Prop===============

(defn ->sgd-config
  ^Sgd
  [{:keys [learning-rate] :or {learning-rate Sgd/DEFAULT_SGD_LR}}]
  (Nesterovs. ^double (double learning-rate)))

;;==================================================
;;=======================META=======================
;;==================================================

(def gradient-update-configs
  {:ams-grad    ->ams-grad-config
   :adadelta    ->ada-delta-config
   :adagrad     ->ada-grad-config
   :adamax      ->ada-max-config
   :adam        ->adam-config
   :nadam       ->nadam-config
   :nesterovs   ->nesterovs-config
   :rms-prop    ->rms-prop-config
   :sgd         ->sgd-config})

;;======================================================
;;=======================BUILDERS=======================
;;======================================================

;;=================Built-in===============

(defn ->gradient-updater-config
  "Builds a gradient updater configuration (IUpdater)
   from updater-kind and specific arguments.
   Basically requires two arguments, with both being
   able to be nested into a single map.
   Throws an exception if the updater does not exist.
   See documentation
   Input :
   - kind : gradient updater kind as a keyword
   - options : gradient updater specific configuration
   Usage :
   (->gradient-updater-config :ada-delta {:rho 1.256})
   ~
   (->gradient-updater-config {:kind :ada-delta , options {:rho 1.256}})"
  ^IUpdater
  ([{:keys [type] :as options}]
   (->gradient-updater-config type options))
  ([type options]
   (if-let [config-fn (type gradient-update-configs)]
     (config-fn options)
     (throw (Exception. (str "GRADIENT UPDATER - Unknown type : " type))))))


;; Add Clojure interface for fast prototype

;;========================================================
;;=======================PREDICATES=======================
;;========================================================

;;=================AMS grad===============

(defn ams-grad-config?
  [obj]
  (instance? AMSGrad obj))

;;=================Ada delta===============

(defn ada-delta-config?
  [obj]
  (instance? AdaDelta obj))

;;=================Ada grad===============

(defn ada-grad-config?
  [obj]
  (instance? AdaGrad obj))

;;=================Ada max===============

(defn ada-max-config?
  [obj]
  (instance? AdaMax obj))

;;=================Adam===============

(defn adam-config?
  [obj]
  (instance? Adam obj))

;;=================Nadam===============

(defn nadam-config?
  [obj]
  (instance? Nadam obj))

;;=================Nesterovs===============

(defn nesterovs-config?
  [obj]
  (instance? Nesterovs obj))

;;=================RMS Prop===============

(defn rms-prop-config?
  [obj]
  (instance? RmsProp obj))

;;=================Sgd===============

(defn sgd-config?
  [obj]
  (instance? Sgd obj))

;;////////////////////////////////////////////////////////////////////////////
;;============================================================================
;;                                  UPDATERS
;;============================================================================
;;\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

;; Not required to design simple neural networks

;;======================================================
;;=======================SPECIFIC=======================
;;======================================================

;;=================AMS grad===============

(defn ams-grad-updater
  ^AMSGradUpdater
  [^AMSGrad config]
  (AMSGradUpdater. config))

(defn ->ams-grad-updater
  ^AMSGradUpdater
  [obj]
  (ams-grad-updater
    (if (ams-grad-config? obj)
      obj
      (->ams-grad-config obj))))

;;=================Ada delta===============

(defn ada-delta-updater
  ^AdaDeltaUpdater
  [^AdaDelta config]
  (AdaDeltaUpdater. config))

(defn ->ada-delta-updater
  ^AdaDeltaUpdater
  [obj]
  (ada-delta-updater
    (if (ada-delta-config? obj)
      obj
      (->ada-delta-config obj))))

;;=================Ada grad===============

(defn ada-grad-updater
  ^AdaGradUpdater
  [^AdaDelta config]
  (AdaGradUpdater. config))

(defn ->ada-grad-updater
  ^AdaGradUpdater
  [obj]
  (ada-grad-updater
    (if (ada-grad-config? obj)
      obj
      (->ada-grad-config obj))))

;;=================Ada max===============

(defn ada-max-updater
  ^AdaMaxUpdater
  [^AdaDelta config]
  (AdaMaxUpdater. config))

(defn ->ada-max-updater
  ^AdaMaxUpdater
  [obj]
  (ada-max-updater
    (if (ada-max-config? obj)
      obj
      (->ada-max-config obj))))

;;=================Adam===============

(defn adam-updater
  ^AdamUpdater
  [^AdaDelta config]
  (AdamUpdater. config))

(defn ->adam-updater
  ^AdamUpdater
  [obj]
  (adam-updater
    (if (adam-config? obj)
      obj
      (->adam-config obj))))

;;=================Nadam===============

(defn nadam-updater
  ^NadamUpdater
  [^AdaDelta config]
  (NadamUpdater. config))

(defn ->nadam-updater
  ^NadamUpdater
  [obj]
  (nadam-updater
    (if (nadam-config? obj)
      obj
      (->nadam-config obj))))

;;=================Nesterovs===============

(defn nesterovs-updater
  ^NesterovsUpdater
  [^AdaDelta config]
  (NesterovsUpdater. config))

(defn ->nesterovs-updater
  ^NesterovsUpdater
  [obj]
  (nesterovs-updater
    (if (nesterovs-config? obj)
      obj
      (->nesterovs-config obj))))

;;=================Rms prop===============

(defn rms-prop-updater
  ^RmsPropUpdater
  [^AdaDelta config]
  (RmsPropUpdater. config))

(defn ->rms-prop-updater
  ^NadamUpdater
  [obj]
  (rms-prop-updater
    (if (rms-prop-config? obj)
      obj
      (->rms-prop-config obj))))

;;=================Sgd===============

(defn sgd-updater
  ^SgdUpdater
  [^AdaDelta config]
  (SgdUpdater. config))

(defn ->sgd-updater
  ^SgdUpdater
  [obj]
  (sgd-updater
    (if (sgd-config? obj)
      obj
      (->sgd-config obj))))

;;==================================================
;;=======================META=======================
;;==================================================

(def gradient-updaters
  {:ams-grad    ->ams-grad-updater
   :adadelta    ->ada-delta-updater
   :adagrad     ->ada-grad-updater
   :adamax      ->ada-max-updater
   :adam        ->adam-updater
   :nadam       ->nadam-updater
   :nesterovs   ->nesterovs-updater
   :rms-prop    ->rms-prop-updater
   :sgd         ->sgd-updater})

;;======================================================
;;=======================BUILDERS=======================
;;======================================================

;;=================Built-in===============

(defn ->gradient-updater
  ^GradientUpdater
  ([{:keys [type] :as options}]
   (->gradient-updater type options))
  ([type options]
   (if-let [builder (get gradient-updaters type)]
     (builder options)
     (throw (Exception. (str "GRADIENT UPDATER - Unknown type : " type))))))


;;=====================================================
;;=======================METHODS=======================
;;=====================================================

;;=================Config===============

(defn get-config
  ^IUpdater
  [^GradientUpdater obj]
  (.getConfig obj))

;;=================State===============

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

;;=================Apply===============

(defn apply-updater!
  [^GradientUpdater obj gradient iteration epoch]
  (.applyUpdater obj (nda/->nd-array gradient) ^int (int iteration) ^int (int epoch)))

(defn apply-updater
  ^GradientUpdater
  [obj gradient iteration epoch]
  (apply-updater! obj gradient iteration epoch)
  obj)

