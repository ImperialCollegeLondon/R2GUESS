\name{get.sweep.best.model}
\alias{get.sweep.best.model}
\title{Internal function used to generate the regression
  coefficients. This function extracts the sweep(s) for which each
selected models has been visited along the MCMC run.}
\usage{
  get.sweep.best.model(x, models = 1)
}
\arguments{
  \item{x}{an object of class \code{ESS}.}

  \item{models}{a vector containing the rank of the models to look up for.}
}
\value{
  A list containing a Boolean matrix (\code{N.model} rows and
  \code{x\$nsweep} columns), where the element \code{i,j} indicates if
the  \code{i-th} model (ranked according to its decreasing posterior probability) has
been visited at sweep \code{j}.  
}
\description{
Internal function used to generate the regression
  coefficients. This function extracts the sweep(s) at which each
of the top models has been visited along the MCMC run. The rank of
the model(s) to look up for is a parameter of the function.
}

\author{
Benoit Liquet, \email{b.liquet@uq.edu.au},\cr
Marc Chadeau-Hyam \email{m.chadeau@imperial.ac.uk},\cr
Leonardo Bottolo \email{l.bottolo@imperial.ac.uk},\cr
Gianluca Campanella \email{g.campanella11@imperial.ac.uk}
}