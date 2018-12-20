\name{sample.beta}
\alias{sample.beta}
\title{Posterior distribution of the regression coefficients for a chosen model}
\usage{
  sample.beta(x, res.g, Nmonte.sigma = 1, Nmonte = 1)
}
\arguments{
  \item{x}{an object of class \code{ESS}}

  \item{res.g}{an object of class \code{g.prior} produced
  by \code{\link{get.g.sweep}}.}

\item{Nmonte.sigma}{number of re-samples of the posterior variance
    covariance matrix of the outcomes (Sigma), for a given value of
    \code{g} among those observed for the model under investigation.}
  \item{Nmonte}{number of re-samples of the regression coefficient
      vector, for a given value of \code{g} and of the Sigma matrix.}
}

\value{A list containing the sampled values of the regression
  coefficients. Re-samples for a given value of \code{g} among those
  observed for the model under investigation are presented in rows
  (\code{Nmonte x Nmonte.sigma} rows) and columns are arranged such
  that the k-th block of \code{q} values represents the regression
  coefficients of predictor \code{k} for all \code{q} outcomes.}


\description{ The \code{sample.beta} function generates the effect
  size estimates of a chosen model within the best models.}



\examples{
modelY_Hopx <- example.as.ESS.object()
n.sweep <- get.sweep.best.model(modelY_Hopx,models=1:2)
res.g <- get.g.sweep(modelY_Hopx,n.sweep$result,model=1)
beta <- sample.beta(modelY_Hopx,res.g,Nmonte=5,Nmonte.sigma=5)
res.D14Mit3 <- boxplotbeta(modelY_Hopx,beta,variable="D14Mit3")
}

\author{ Benoit Liquet, \email{b.liquet@uq.edu.au},\cr
  Marc Chadeau-Hyam \email{m.chadeau@imperial.ac.uk},\cr Leonardo
  Bottolo \email{l.bottolo@imperial.ac.uk},\cr Gianluca Campanella
  \email{g.campanella11@imperial.ac.uk} }

