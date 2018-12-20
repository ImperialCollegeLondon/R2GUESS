\name{plotmodel}
\alias{plotmodel}
\title{Visualisation of the proximity between best models}
\usage{
  plotmodel(x, threshold.model = 20)
}
\arguments{
  \item{x}{an object of class \code{ESS}.}

  \item{threshold.model}{either an integer representing the number of
      models to be retained in the list of best models, or a value
      defining the minimal model posterior probability for inclusion.}
}
\description{
  The \code{plotmodel} function provides an image plot
  of the proximity (according to an overlap metric) between the best models.
}
\examples{
# load an ESS object
modelY_Hopx <- example.as.ESS.object()
plotmodel(modelY_Hopx,20)
}
\author{
Benoit Liquet, \email{b.liquet@uq.edu.au},\cr
Marc Chadeau-Hyam \email{m.chadeau@imperial.ac.uk},\cr
Leonardo Bottolo \email{l.bottolo@imperial.ac.uk},\cr
Gianluca Campanella \email{g.campanella11@imperial.ac.uk}
}
