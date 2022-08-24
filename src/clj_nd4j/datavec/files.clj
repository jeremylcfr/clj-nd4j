(ns clj-nd4j.datavec.files
  (:import [java.io File InputStream]
           [java.net URL]
           [org.nd4j.common.io ClassPathResource])
  )


;; TO MOVE

(defn file?
  [obj]
  (instance? File obj))

;;////////////////////////////////////////////////////////////////////////////
;;============================================================================
;;                                 BUILDER
;;============================================================================
;;\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

(defn class-path-resource
  "Creates a ClassPathResource object
   which has almost no interest on its
   own apart from trying to extract a file
   from JAR. Used in Datavec if explicitely
   told. ClassLoader arity is not provided for
   now, will see if need arises"
  ^ClassPathResource
  [^String path]
  (ClassPathResource. path))

;;////////////////////////////////////////////////////////////////////////////
;;============================================================================
;;                                 ACCESSORS
;;============================================================================
;;\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

;;=========================================================
;;=======================I/O OBJECTS=======================
;;=========================================================

;;=================File===============

(defn get-file
  ^File
  [^ClassPathResource c]
  (.getFile c))

;;=================Input stream===============

(defn get-input-stream
  ^InputStream
  [^ClassPathResource c]
  (.getInputStream c))

;;===================================================
;;=======================PATHS=======================
;;===================================================

;;=================URL===============

(defn get-url
  ^URL
  [^ClassPathResource c]
  (.getURL c))

;;=================Filename===============

(defn get-filename
  ^String
  [^ClassPathResource c]
  (.getFilename c))

;;////////////////////////////////////////////////////////////////////////////
;;============================================================================
;;                                  PREDICATES
;;============================================================================
;;\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

;;==================================================
;;=======================TYPE=======================
;;==================================================

(defn class-path-resource?
  [obj]
  (instance? ClassPathResource obj))

;;=====================================================
;;=======================METHODS=======================
;;=====================================================

(defn resource-exists?
  [^ClassPathResource c]
  (.exists c))

;;////////////////////////////////////////////////////////////////////////////
;;============================================================================
;;                                 COERCERS
;;============================================================================
;;\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

(defn ->class-path-resource
  [obj]
  ^ClassPathResource
  (cond (class-path-resource? obj)
          obj
        (string? obj)
          (class-path-resource obj)
        (file? obj)
          (class-path-resource (.getAbsolutePath ^File obj))
        :else
          (throw (Exception. (str "ClassPathResource - Input : " obj " is not coercible to a class path resource")))))

;;////////////////////////////////////////////////////////////////////////////
;;============================================================================
;;                                QUICK BUILDER
;;============================================================================
;;\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

(defn ->file-from-jar
  "Function to use to load a File
   from ND4J API (which tries to
   extract files from JAR also).
   Not really useful on its own
   in the context of this library.
   In pratice, you will see it as
   an option in clj-datavec"
  ^File
  [obj]
  (get-file (->class-path-resource obj)))

