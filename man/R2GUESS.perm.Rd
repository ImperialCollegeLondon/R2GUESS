\name{R2GUESS.perm}
\alias{R2GUESS.perm}
\title{Internal function used for the parallel computation on the permutation procedure for the calibration of the FDR threshold
regarding the significance of the MPPI. This function run GUESS on permutation sample.}
\usage{
  R2GUESS.perm(dataY, dataX, path.inputx, path.inputy,
    path.output, path.par, path.init = NULL, file.par,
    file.init = NULL, file.log = NULL, nsweep,
    burn.in = 1000, Egam = 2, Sgam = 2, root.file.output,
    time = TRUE, top = 100, history = TRUE, label.X = NULL,
    label.Y = NULL, nb.chain, conf = 0, cuda = TRUE,
    MAP.file = NULL, p, q, n,time.limit=NULL,seed=NULL)
}
\description{
  Internal function used for the parallel computation on
  the permutation procedure for the calibration of the FDR
  threshold regarding the significance of the MPPI. This
  function run GUESS on permutation sample.
}
\keyword{internal}

