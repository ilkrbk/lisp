(import [pandas :as pd])

(setv path "./minimax-statistic/data.csv")
(setv data (pd.read_csv path))

(setv time (get data "time"))
(print "Expected value:" (.mean time))

(setv score (get data "score"))
(print "Dispersion:" (.var score))