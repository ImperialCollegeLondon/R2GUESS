\name{plotcim.explore}
\alias{plotcim.explore}
\title{Plots a cluster image mapping of correlations between
  outcomes and all predictors}
\usage{
  plotcim.explore(matX, matY)
}
\arguments{
  \item{matX}{data frame corresponding to the predictors.}

  \item{matY}{data frame corresponding to the outcomes.}
}
\description{
  The \code{plot.cim.explore} function plots a cluster image
  mapping of correlations between outcomes and all the  predictors.
}
\details{
  To be used with a small number of predictors (<1,000).
}
\examples{
data(data.Y.Hopx)
data(data.X)
plotcim.explore(data.X,data.Y.Hopx)
}
\author{
Benoit Liquet, \email{b.liquet@uq.edu.au},\cr
Marc Chadeau-Hyam \email{m.chadeau@imperial.ac.uk},\cr
Leonardo Bottolo \email{l.bottolo@imperial.ac.uk},\cr
Gianluca Campanella \email{g.campanella11@imperial.ac.uk}
}
