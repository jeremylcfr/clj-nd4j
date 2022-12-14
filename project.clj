(defproject io.github.jeremylcfr/clj-nd4j "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Apache License"
            :url "https://www.apache.org/licenses/LICENSE-2.0"}
  :dependencies [[org.clojure/clojure "1.11.1"]
                 [org.clojure/algo.generic "0.1.3"]
                 [io.github.jeremylcfr/clj-java-commons "1.0.0"]
                 [org.nd4j/nd4j-api "1.0.0-M2.1"]
                 [org.nd4j/nd4j-common "1.0.0-M2.1"]
                 [org.nd4j/nd4j-native "1.0.0-M2.1"]] 
  :repositories [["github" {:url "https://maven.pkg.github.com/jeremylcfr/clj-nd4j"
                            :username "jeremylcfr" 
                            :password :env/GITHUB_TOKEN 
                            :sign-releases false}]]
  :scm {:name "git" :url "https://github.com/jeremylcfr/clj-nd4j"}
  :profiles {:dev {:source-paths ["repl"]}}
  :repl-options {:init-ns repl})
