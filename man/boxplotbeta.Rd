\name{boxplotbeta}
\alias{boxplotbeta}
\title{Draws boxplots of the posterior distribution of regression
  coefficient(s) for a given predictor}
\usage{
  boxplotbeta(x, beta, variable)
}
\arguments{
  \item{x}{an object of class \code{ESS}}

  \item{beta}{a list produced by
  \code{\link{sample.beta}} representing the regression coefficient of
the predictor of interest with respect to the outcome(s) in the response
matrix.}

\item{variable}{name of the predictor of interest, or its position in
    \code{dataX}.}

}


\value{Returns a matrix  containing samples (iterations in rows) of the regression
  coefficients (\code{q} columns) corresponding to the variable specified by the argument
  \code{variable} with respect to all outcomes.}


\description{ The
  \code{boxplot.beta} function draws a boxplot representation of the
  posterior distribution of regression coefficient(s) for a
  user-defined predictor. The function requires a preliminary run of
  \code{\link{sample.beta}}.}

\author{ Benoit Liquet,
  \email{b.liquet@uq.edu.au},\cr Marc Chadeau-Hyam
  \email{m.chadeau@imperial.ac.uk},\cr Leonardo Bottolo
  \email{l.bottolo@imperial.ac.uk},\cr Gianluca Campanella
  \email{g.campanella11@imperial.ac.uk} }

\seealso{ {\code{\link{sample.beta}}}}

