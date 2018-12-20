\name{plot.ESS}
\alias{plot.ESS}
\title{Provides diagnostic plots to assess the convergence of the MCMC
procedure along the run}
\usage{
  \method{plot}{ESS} (x, ...)
}
\arguments{
  \item{x}{an object of class \code{ESS}.}

  \item{...}{additional arguments: \code{range.SF} range
  value for the shrinkage factor plot (\code{g});
  \code{range.MS} range value for the model size plot;
  \code{range.LP} range value for the log posterior plot.}
}
\description{
  The \code{plot.ESS} function provides a set of four plots
  summarising the behaviour of all chains along the Evolutionary Monte
  Carlo run. These plots are used to assess the convergence of
  the algorithm and comprise: (i)- the history plot of the shrinkage factor
  \code{g}, (ii)- the history plot of the model size in each chain,
  (iii)- the history plot of the log posterior in each chain, and (iv)-
  the history plot of each chain's temperature (during the burn-in
  only, as temperatures are fixed at the end of the burn-in).
}
\examples{
# load an ESS object
modelY_Hopx <- example.as.ESS.object()
plot(modelY_Hopx)
}
\author{
Benoit Liquet, \email{b.liquet@uq.edu.au},\cr
Marc Chadeau-Hyam \email{m.chadeau@imperial.ac.uk},\cr
Leonardo Bottolo \email{l.bottolo@imperial.ac.uk},\cr
Gianluca Campanella \email{g.campanella11@imperial.ac.uk}
}
