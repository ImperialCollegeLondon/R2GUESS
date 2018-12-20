\name{print.ESS}
\alias{print.ESS}
\title{Provides a \code{print} method for class \code{ESS}}
\usage{
  \method{print}{ESS} (x, ...)
}
\arguments{
  \item{x}{an object of class \code{ESS}}

  \item{...}{not used currently}
}
\description{
  Print method for R2GUESS
}
\details{
  \code{print} method for \code{ESS} class, returns a
  description of the \code{x} object. 
}
\examples{
# load an ESS object
modelY_Hopx <- example.as.ESS.object()
print(modelY_Hopx)
}
\author{
Benoit Liquet, \email{b.liquet@uq.edu.au},\cr
Marc Chadeau-Hyam \email{m.chadeau@imperial.ac.uk},\cr
Leonardo Bottolo \email{l.bottolo@imperial.ac.uk},\cr
Gianluca Campanella \email{g.campanella11@imperial.ac.uk}
}

\seealso{
\code{\link{as.ESS.object}}
}
