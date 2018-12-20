\name{plotvariable}
\alias{plotvariable}
\title{Visualisation of the best models}
\usage{
  plotvariable(x, threshold.model = 0.05, file.annotation = NULL)
}
\arguments{
  \item{x}{an object of class \code{ESS}.}

  \item{threshold.model}{either an integer representing the number of
      model to be retained in the list of best models, or a value
      defining the minimal model posterior probability.}

  \item{file.annotation}{text file containing the name of the
      predictors. This file is optional if a MAP.file has been specified.}
}
\value{
  A list of predictors included in the best models.
}
\description{
  The \code{plotvariable} function provides a plot indicating the variables
  included in each of the best models.
}
\examples{
# load an ESS object
modelY_Hopx <- example.as.ESS.object()
plotvariable(modelY_Hopx,20)
}
\author{
Benoit Liquet, \email{b.liquet@uq.edu.au},\cr
Marc Chadeau-Hyam \email{m.chadeau@imperial.ac.uk},\cr
Leonardo Bottolo \email{l.bottolo@imperial.ac.uk},\cr
Gianluca Campanella \email{g.campanella11@imperial.ac.uk}
}
