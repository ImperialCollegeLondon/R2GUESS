\name{plotcim}
\alias{plotcim}
\title{Clustered Image Maps (CIMs) (heat maps)}
\usage{
  plotcim(x, select.variable, labelX = NULL)
}
\arguments{
  \item{x}{an object of class \code{ESS}.}

  \item{select.variable}{name of the selected predictors.}

  \item{labelX}{names of the predictors used when running
  \code{\link{R2GUESS}}. This argument is optional if a \code{\link{MAP.file}} has been provided.}
}
\description{
  This function generates color-coded Clustered Image Maps
  (CIMs) representing the pairwise
  correlation between a subset of predictors and the outcomes.
}
\examples{
modelY_Hopx <- example.as.ESS.object()
MPPI.Hopx <-
plotMPPI(modelY_Hopx,threshold.model=20,threshold.variable=0.25,Figure=FALSE)
plotcim(modelY_Hopx,select.variable=MPPI.Hopx$var.TOP.MPI)
}
\author{
Benoit Liquet, \email{b.liquet@uq.edu.au},\cr
Marc Chadeau-Hyam \email{m.chadeau@imperial.ac.uk},\cr
Leonardo Bottolo \email{l.bottolo@imperial.ac.uk},\cr
Gianluca Campanella \email{g.campanella11@imperial.ac.uk}
}
