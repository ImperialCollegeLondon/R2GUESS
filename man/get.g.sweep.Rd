\name{get.g.sweep}
\alias{get.g.sweep}
\title{Internal function used to generate the regression
  coefficients. This function extracts the values of the shrinkage
  factor g for a given model specified by its ranked posterior probability}
\usage{
  get.g.sweep(x, res, model)
}
\arguments{
  \item{x}{an object of class \code{ESS}.}

  \item{res}{a matrix of boolean values indicating at which sweeps of
      the MCMC run the model has been visited. This matrix is the
      output of \code{\link{get.sweep.best.model}}.
}

  \item{model}{integer specifying the rank (according to the model posterior
      probability) of the model under investigation.}
}




\value{ An object of class \code{g.prior} containing: 


\item{g.value}{the observed values of \code{g} for the chosen model.}
\item{predictors}{the variables belonging to the model considered.}
\item{X}{the X design matrix for the model.}
\item{Y}{the centred outcome matrix.}}

\description{Internal function used for the generation of the
  regression coefficients beta. \code{get.g.sweep} extracts the values
  of the shrinkage factor \code{g} associated with the sweeps at which a
  given model has been visited. The model under investigation is
  defined by its rank in terms of posterior probability.  }

\author{
Benoit Liquet, \email{b.liquet@uq.edu.au},\cr
Marc Chadeau-Hyam \email{m.chadeau@imperial.ac.uk},\cr
Leonardo Bottolo \email{l.bottolo@imperial.ac.uk},\cr
Gianluca Campanella \email{g.campanella11@imperial.ac.uk}
}

\seealso{

\code{\link{get.sweep.best.model}}
}