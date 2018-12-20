\name{Extend.R2GUESS}
\alias{Extend.R2GUESS}
\title{Extends an already finished \code{R2GUESS} run for an extra
  user-defined number of iterations}
\usage{
  Extend.R2GUESS(x, niter, time.limit=NULL)
}
\arguments{
  \item{x}{an object of class \code{ESS}.}

  \item{niter}{integer specifying the number of additional sweeps to
      run the model for.}

  \item{time.limit}{numeric representing the maximum computation time (in
      hours) allowed for extending the run. By default (\code{NULL}),
      the run will continue until completion.}
}

\value{
  An object of class \code{ESS} whose fields are detailed in
  \code{\link{as.ESS.object}}, and which contains results from the
  additional sweeps.}


\description{ The \code{Extend.R2GUESS} extends an already finished
  \code{R2GUESS} run.}



\details{ The \code{Extend.R2GUESS} extends an already finished
  \code{R2GUESS} run for an additional \code{niter} sweeps. Results
  from these extra iterations will be appended to the existing history
  files, and posterior calculations (MPPI, MPP, and the list of best models) will be entirely
  updated. These additional iterations will use the same pseudo-random
  number generator as the one used for the initial run, and will
  initialise it at the state it was while finishing the original
  run. If the previous run was interrupted before reaching the
  user-defined \code{time.limit}, \code{Extend.R2GUESS} will return an
  error message.}


\examples{
\dontrun{
modelY_Hopx <- example.as.ESS.object()
## Be careful the Output files will be created in modelY_Hopx$path.output
## If you want to save the new Ouput copy/paste the 
## folder modelY_Hopx$out in an appropriate place
## and change modelY_Hopx$path.output
modelY_Hopx_extend <- Extend.R2GUESS(modelY_Hopx,100)
}
}
\author{
Benoit Liquet, \email{b.liquet@uq.edu.au},\cr
Marc Chadeau-Hyam \email{m.chadeau@imperial.ac.uk},\cr
Leonardo Bottolo \email{l.bottolo@imperial.ac.uk},\cr
Gianluca Campanella \email{g.campanella11@imperial.ac.uk}
}
\seealso{
\code{\link{Resume.R2GUESS}, \link{Postprocess.R2GUESS}, \link{as.ESS.object}}}


