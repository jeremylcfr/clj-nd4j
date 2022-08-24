(ns clj-nd4j.ml.loss
  (:import [org.nd4j.linalg.lossfunctions ILossFunction LossFunctions LossFunctions$LossFunction]))

;;////////////////////////////////////////////////////////////////////////////
;;============================================================================
;;                                CONFIGURATION
;;============================================================================
;;\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


(def loss-fns-pointers
  {:mse                            LossFunctions$LossFunction/MSE
   :l1                             LossFunctions$LossFunction/L1
   :xent                           LossFunctions$LossFunction/XENT
   :mc-xent                        LossFunctions$LossFunction/MCXENT
   :squared-loss                   LossFunctions$LossFunction/SQUARED_LOSS
   :reconstruction-cross-entropy   LossFunctions$LossFunction/RECONSTRUCTION_CROSSENTROPY
   :negative-log-likehood          LossFunctions$LossFunction/NEGATIVELOGLIKELIHOOD
   :cosine-proximity               LossFunctions$LossFunction/COSINE_PROXIMITY
   :hinge                          LossFunctions$LossFunction/HINGE
   :squared-hinge                  LossFunctions$LossFunction/SQUARED_HINGE
   :kl-divergence                  LossFunctions$LossFunction/KL_DIVERGENCE
   :mae                            LossFunctions$LossFunction/MEAN_ABSOLUTE_ERROR
   :l2                             LossFunctions$LossFunction/L2
   :mape                           LossFunctions$LossFunction/MEAN_ABSOLUTE_PERCENTAGE_ERROR
   :log-rmse                       LossFunctions$LossFunction/MEAN_SQUARED_LOGARITHMIC_ERROR
   :poisson                        LossFunctions$LossFunction/POISSON})


(defn ->loss-fn
  "Returns a loss function from its keyword identifier.
   Throws an exception if the function does not exist (i.e.
   not built-in).
   See documentation.
   Input :
   - kind : keyword referencing the loss function to use
   Usage :
   (->loss-fn :mae)"
  ^ILossFunction
  [kind]
  (if-let [f (kind loss-fns-pointers)]
    f
    (throw (Exception. (str "LOSS - Unknown loss function type : " kind)))))


