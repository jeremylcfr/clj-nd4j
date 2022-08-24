(ns clj-nd4j.ml.activation
  (:import [org.nd4j.linalg.activations IActivation Activation]))

;;////////////////////////////////////////////////////////////////////////////
;;============================================================================
;;                                CONFIGURATION
;;============================================================================
;;\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


(def activation-fns-pointers
  {:cube            Activation/CUBE
   :elu             Activation/ELU
   :hard-sigmoid    Activation/HARDSIGMOID
   :hard-tanh       Activation/HARDTANH
   :identity        Activation/IDENTITY
   :leaky-relu      Activation/LEAKYRELU
   :rational-tanh   Activation/RATIONALTANH
   :relu            Activation/RELU
   :rrelu           Activation/RRELU
   :sigmoid         Activation/SIGMOID
   :softmax         Activation/SOFTMAX
   :softplus        Activation/SOFTPLUS
   :softsign        Activation/SOFTSIGN
   :tanh            Activation/TANH
   :rectified-tanh  Activation/RECTIFIEDTANH
   :selu            Activation/SELU
   :swish           Activation/SWISH})

(defn activation-fn?
  [obj]
  (instance? Activation obj))

(defn ->activation-fn
  "Returns an activation function from its
   keyword. Throws an exception if the function
   does not exist.
   See documentation
   Input :
   - kind : a keyword referencing the activation function to use
   Usage :
   (->activation-fn :cube)"
  ^Activation
  [kind]
  (if (activation-fn? kind)
    kind
    (if-let [f (kind activation-fns-pointers)]
      f
      (throw (Exception. (str "ACTIVATION - Unknown activation function type : " kind))))))

;; TODO : Add transforms


