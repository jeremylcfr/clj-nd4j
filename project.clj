(defproject jeremylcfr/clj-nd4j "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.11.1"]
                 [org.clojure/algo.generic "0.1.3"]
                 [jeremylcfr/clj-java-commons "0.1.0-SNAPSHOT"]
                 [org.nd4j/nd4j-api "1.0.0-M2.1"]
                 [org.nd4j/nd4j-common "1.0.0-M2.1"]
                 [org.nd4j/nd4j-native "1.0.0-M2.1"]]
  :profiles {:dev {:source-paths ["repl"]}}
  :repl-options {:init-ns repl})
