\name{summary.ESS}
\alias{summary.ESS}
\title{\code{summary} method for class \code{ESS}}
\usage{
  \method{summary}{ESS} (object, ...)
}
\arguments{
  \item{object}{an object of class \code{ESS}.}

  \item{...}{integer indicating the number of best models
  to be summarized.}
}
\description{
  Summary method for \code{ESS} objects.
}
\details{\code{summary} method for the \code{ESS} class. It returns a
  summary of the \code{x} object including the list of the
  best models visited}
\value{ The function \code{summary} returns a data frame containing
  the following fields describing all of the best models according to:
\item{Rank}{the rank of the model according to its posterior probability.}
\item{nVisits}{the number of times the model
 has been visited during the run (including the
 burn-in).}
\item{FirstVisit}{the iteration at which the model was first visited.}
\item{nEvalBefore1st}{the number of iteration before the first visit
    of the model.}
\item{ModeSize}{the number of variables in the model.}
\item{logCondPost}{the log
posterior probability (integrated over the shrinkage factor g).}
\item{postProb}{the posterior probability of the model.}
\item{jeffrey}{the Jeffrey's scale for the model.}
\item{modelName}{the list of variables (predictors) included in the model.}
}

\examples{
# load an ESS object
modelY_Hopx <- example.as.ESS.object()
summary(modelY_Hopx,20)
}
\author{
Benoit Liquet, \email{b.liquet@uq.edu.au},\cr
Marc Chadeau-Hyam \email{m.chadeau@imperial.ac.uk},\cr
Leonardo Bottolo \email{l.bottolo@imperial.ac.uk},\cr
Gianluca Campanella \email{g.campanella11@imperial.ac.uk}
}
